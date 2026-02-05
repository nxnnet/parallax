"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the following services:
    1.Executor each tp_rank as a subprocess.
    2.HTTP server as a subprocess.
    3.P2P server as a subprocess.

Example command:
python src/parallax/launch.py \
    --model-path Qwen/Qwen3-0.6B \
    --max-num-tokens-per-batch 16384 \
    --max-batch-size 128 \
    --start-layer 0 \
    --end-layer 28
"""

import argparse
import multiprocessing
import os
import tempfile
import time

from parallax.p2p.server import ServerState, launch_p2p_server_process, stop_p2p_server
from parallax.server.executor.factory import run_executor_process, stop_executor_process
from parallax.server.http_server import launch_http_server, stop_http_server
from parallax.server.server_args import parse_args
from parallax.utils.shared_state import SharedState
from parallax.utils.utils import fetch_model_from_hf, initialize_nccl_port
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

logger = get_logger("parallax.launch")


def _update_args_from_shared_state(args, shared_state: SharedState, force_update: bool):
    """Update args with layer allocation from shared state"""
    model_info = shared_state.get_model_info()
    args.start_layer = model_info["block_start_index"]
    args.end_layer = model_info["block_end_index"]
    if args.model_path is not None and force_update == False:
        # Use local model path first
        pass
    elif model_info["model_name"]:
        # Update model_path if provided
        args.model_path = model_info["model_name"]
        logger.debug(f"更新模型路径为: {args.model_path}")
    else:
        assert False, "Neither scheduler nor worker provides a valid model path!"
    # Update tp_size if provided, otherwise keep current value
    args.tp_size = model_info["tp_size"] or args.tp_size
    # Update weight refit switch
    args.enable_weight_refit = model_info["enable_weight_refit"] or args.enable_weight_refit
    args.weight_refit_mode = model_info["weight_refit_mode"] or args.weight_refit_mode


def _stop_executor_processes(executor_subprocs):
    """Stop all executor processes"""
    for executor_process in executor_subprocs:
        if executor_process.is_alive():
            logger.debug(f"正在终止 Executor 进程 {executor_process.pid}")
            stop_executor_process(executor_process)


def _wait_executors_check_layer_change(shared_state: SharedState, executor_subprocs):
    """Wait for executor processes and check if layer allocation changed.

    Returns:
        True if layer allocation changed (need to reload executors),
        False if all executors exited normally.
    """
    while any(proc.is_alive() for proc in executor_subprocs):
        for proc in executor_subprocs:
            if proc.is_alive():
                proc.join(timeout=1.0)  # Check every second

        if shared_state.get_layer_allocation_changed():
            return True

    # Check race condition: layer allocation changed after all processes exited
    return shared_state.get_layer_allocation_changed()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    p2p_server_process = None
    http_server_process = None
    executor_subprocs = []
    # Shared state for layer allocation info (used when P2P server is in subprocess)
    # 用于层分配信息的共享状态（当 P2P 服务器在子进程中运行时使用）
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.JOINING.value)

    try:
        args = parse_args()
        set_log_level(args.log_level)
        logger.debug(f"启动参数: {args}")
        args.recv_from_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.send_to_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_input_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.executor_output_ipc = f"ipc://{tempfile.NamedTemporaryFile().name}"
        if args.nccl_port is None:
            args.nccl_port = initialize_nccl_port()

        # Silence tokenizer warnings
        # 禁用 tokenizer 的并行化警告
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.debug(f"Executor 输入地址: {args.executor_input_ipc}")
        logger.debug(f"Executor 输出地址: {args.executor_output_ipc}")
        logger.debug(f"NCCL 端口: {args.nccl_port}")

        # Pipe for subprocess communication
        # 用于子进程通信的管道
        conn_main, conn_refit = multiprocessing.Pipe()

        if args.scheduler_addr is None:
            # 本地运行模式（无调度器）
            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            config = fetch_model_from_hf(args.model_path, local_files_only=args.use_hfcache)
            if args.start_layer is None:
                args.start_layer = 0
            if args.end_layer is None:
                args.end_layer = config.get("num_hidden_layers")

            # only launch http server on head node
            # 仅在头节点启动 HTTP 服务器
            if args.start_layer == 0:
                http_server_process = launch_http_server(args)
            # Launch P2P server as subprocess
            # 启动 P2P 服务器作为子进程
            if not (args.start_layer == 0 and args.end_layer == config.get("num_hidden_layers")):
                p2p_server_process = launch_p2p_server_process(
                    initial_peers=args.initial_peers,
                    scheduler_addr=args.scheduler_addr,
                    relay_servers=args.relay_servers,
                    pp_start_layer=args.start_layer,
                    pp_end_layer=args.end_layer,
                    hidden_layers=config.get("num_hidden_layers"),
                    tp_size=args.tp_size,
                    dp_size=args.dp_size,
                    tcp_port=args.tcp_port,
                    udp_port=args.udp_port,
                    dht_prefix=args.dht_prefix,
                    announce_maddrs=args.announce_maddrs,
                    http_port=args.port,
                    notify_url=args.notify_url,
                    recv_from_peer_addr=args.recv_from_peer_addr,
                    send_to_peer_addr=args.send_to_peer_addr,
                    model_name=args.model_path,
                    max_batch_size=args.max_batch_size,
                    max_sequence_length=args.max_sequence_length,
                    param_mem_ratio=args.param_mem_ratio,
                    kvcache_mem_ratio=args.kvcache_mem_ratio,
                    shared_state=shared_state.dict,
                    log_level=args.log_level,
                    conn=conn_main,
                    key_path=args.key_path,
                )

            # Build connectors for tp communication
            # 构建用于 TP 通信的连接器
            conn_tp_0 = [conn_refit]
            conn_tp_i = []
            for i in range(1, args.tp_size):
                conn1, conn2 = multiprocessing.Pipe()
                conn_tp_0.append(conn1)
                conn_tp_i.append(conn2)
            # Launch all executor processes (including tp_rank=0)
            # 启动所有 Executor 进程（包括 tp_rank=0）
            for tp_rank in range(args.tp_size):
                args_copy = argparse.Namespace(**vars(args))
                args_copy.tp_rank = tp_rank
                proc = multiprocessing.Process(
                    target=run_executor_process,
                    args=(
                        args_copy,
                        shared_state.dict,  # Pass dict to subprocess
                        conn_tp_0 if tp_rank == 0 else [conn_tp_i[tp_rank - 1]],
                    ),
                )
                proc.start()
                executor_subprocs.append(proc)

            time.sleep(2)  # Give executors time to start
            shared_state.set_status(ServerState.READY.value)

            # Wait for all executor processes
            # 等待所有 Executor 进程
            for proc in executor_subprocs:
                proc.join()
        else:
            # 分布式模式（加入集群）
            # Launch P2P server as subprocess (with scheduler)
            # 启动 P2P 服务器作为子进程（连接到调度器）
            # Pass dict to subprocess (multiprocessing requires serializable objects)
            p2p_server_process = launch_p2p_server_process(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=args.start_layer,
                pp_end_layer=args.end_layer,
                hidden_layers=None,
                tp_size=args.tp_size,
                dp_size=args.dp_size,
                tcp_port=args.tcp_port,
                udp_port=args.udp_port,
                dht_prefix=args.dht_prefix,
                announce_maddrs=args.announce_maddrs,
                http_port=args.port,
                notify_url=args.notify_url,
                recv_from_peer_addr=args.recv_from_peer_addr,
                send_to_peer_addr=args.send_to_peer_addr,
                model_name=args.model_path,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
                param_mem_ratio=args.param_mem_ratio,
                kvcache_mem_ratio=args.kvcache_mem_ratio,
                shared_state=shared_state.dict,  # Pass dict to subprocess
                log_level=args.log_level,
                conn=conn_main,
                key_path=args.key_path,
            )

            # Wait for layer allocation from scheduler (via shared state)
            # 等待调度器分配层信息（通过共享状态）
            logger.debug("正在等待调度器分配层信息...")
            max_wait_time = 300  # 5 minutes
            wait_start = time.time()
            while True:
                model_info = shared_state.get_model_info()
                if (
                    model_info["block_start_index"] is not None
                    and model_info["block_end_index"] is not None
                    and model_info["model_name"] is not None
                ):
                    break
                if time.time() - wait_start > max_wait_time:
                    logger.error("Timeout waiting for layer allocation from scheduler")
                    raise RuntimeError("Failed to get layer allocation from scheduler")
                time.sleep(1)

            # Get layer allocation from shared state
            # 从共享状态获取层分配信息
            _update_args_from_shared_state(args, shared_state, force_update=False)

            logger.debug(
                f"启动 Executor，起始层: {args.start_layer}, 结束层: {args.end_layer}, "
                f"模型: {args.model_path}"
            )

            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            # Main execution loop with layer reallocation support
            # 主执行循环，支持层重新分配
            while True:
                try:
                    # only launch http server on head node
                    # 仅在头节点启动 HTTP 服务器
                    if args.start_layer == 0:
                        http_server_process = launch_http_server(args)

                    # Build connectors for tp communication
                    # 构建用于 TP 通信的连接器
                    conn_tp_0 = [conn_refit]
                    conn_tp_i = []
                    for i in range(1, args.tp_size):
                        conn1, conn2 = multiprocessing.Pipe()
                        conn_tp_0.append(conn1)
                        conn_tp_i.append(conn2)
                    # Launch all executor processes (including tp_rank=0)
                    # 启动所有 Executor 进程（包括 tp_rank=0）
                    executor_subprocs = []
                    for tp_rank in range(args.tp_size):
                        args_copy = argparse.Namespace(**vars(args))
                        args_copy.tp_rank = tp_rank
                        proc = multiprocessing.Process(
                            target=run_executor_process,
                            args=(
                                args_copy,
                                shared_state.dict,  # Pass dict to subprocess
                                conn_tp_0 if tp_rank == 0 else [conn_tp_i[tp_rank - 1]],
                            ),
                        )
                        proc.start()
                        executor_subprocs.append(proc)

                    # Wait for executors and restart if layer allocation changes
                    # 等待 Executor 运行，并检查层分配是否发生变化
                    if _wait_executors_check_layer_change(shared_state, executor_subprocs):
                        logger.warning("Layer allocation changed! Stopping executors to reload...")
                        # Reset flag and set status to INITIALIZING
                        # 重置标志并将状态设置为 INITIALIZING
                        shared_state.update(
                            _layer_allocation_changed=False,
                            status=ServerState.INITIALIZING.value,
                        )
                        _stop_executor_processes(executor_subprocs)
                        if http_server_process is not None:
                            stop_http_server(http_server_process)
                        _update_args_from_shared_state(args, shared_state, force_update=True)
                        logger.info(
                            f"Reloading executor with layers [{args.start_layer}, {args.end_layer})"
                        )
                        continue

                    # All processes exited normally
                    # 所有进程正常退出
                    break
                except KeyboardInterrupt:
                    logger.debug("接收到中断信号，正在关闭...")
                    break
                except Exception as e:
                    logger.exception(f"Executor error: {e}")
                    # Shutdown all executor processes on error
                    # 发生错误时关闭所有 Executor 进程
                    for proc in executor_subprocs:
                        if proc.is_alive():
                            stop_executor_process(proc)
                    raise
    except KeyboardInterrupt:
        logger.debug("接收到中断信号，正在关闭...")
    except Exception as e:
        logger.exception(e)
    finally:
        # Shutdown all processes
        logger.debug("正在关闭所有进程...")

        # Shutdown executor subprocesses
        # 关闭 Executor 子进程
        for executor_process in executor_subprocs:
            if executor_process.is_alive():
                stop_executor_process(executor_process)

        # Shutdown P2P server subprocess
        # 关闭 P2P 服务器子进程
        if p2p_server_process is not None:
            stop_p2p_server(p2p_server_process)

        # Shutdown http server
        # 关闭 HTTP 服务器
        if http_server_process is not None:
            stop_http_server(http_server_process)

        logger.debug("所有进程已关闭。")
