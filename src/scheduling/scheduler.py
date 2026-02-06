"""
Scheduler for Layer Allocation and Request Routing.
负责层分配(Layer Allocation)和请求路由(Request Routing)的调度器。
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.layer_allocation import (
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
)
from scheduling.model_info import ModelInfo
from scheduling.node import Node, RequestSignal
from scheduling.node_management import NodeManager
from scheduling.request_routing import (
    DynamicProgrammingRouting,
    RandomizedOverDynamicPipelinesRouting,
    RoundRobinOverFixedPipelinesRouting,
)

logger = get_logger(__name__)


class Scheduler:
    """Coordinates allocation, node materialization, and request routing.
    协调层分配、节点具体化和请求路由的核心类。
    主要职责：
    1. 管理节点生命周期（加入、离开、更新）。
    2. 使用分配器（LayerAllocator）将模型层分配给节点。
    3. 使用路由器（RequestRouter）为每个请求规划最佳路径。
    4. 监控节点健康状态（心跳检测）。
    """

    def __init__(
        self,
        model_info: ModelInfo,
        nodes: List[Node],
        min_nodes_bootstrapping: int = 1,
        enable_weight_refit: bool = False,
        weight_refit_mode: str = "disk",
        strategy: Literal["greedy", "dp"] = "dp",
        routing_strategy: Literal["rr", "dp"] = "rr",
        *,
        request_arrival_horizon_sec: float = 600.0,
        rebalance_threshold: float = float("inf"),
        water_filling_max_iterations: int = 40,
        heartbeat_timeout: float = 30.0,
        trim_layers_on_turning_points: bool = True,
    ) -> None:
        """Initialize the scheduler.
        初始化调度器。

        Args:
            model_info: 模型架构信息，供分配器和路由器使用。
            nodes: 初始候选节点列表。
            min_nodes_bootstrapping: 尝试初始分配所需的最少节点数。
            strategy: 层分配策略（动态规划(dp)或贪婪算法(greedy)）。
            routing_strategy: 请求路由策略（"dp" 为动态规划，"greedy" 为轮询完整流水线并跳过过载节点）。
            request_arrival_horizon_sec: 跟踪到达率的滑动窗口时间范围（秒）。
            rebalance_threshold: 触发重新平衡分配的阈值。
            water_filling_max_iterations: 注水算法分配的最大迭代次数。
            request_warm_up_for_reshard: 用于检测截断点的预热请求数量。
            heartbeat_timeout: 判定节点心跳过期的超时时间（秒）。
        """
        self.model_info = model_info
        self.num_layers = model_info.num_layers
        self.routing_strategy: Literal["rr", "dp"] = routing_strategy
        self.enable_weight_refit = enable_weight_refit
        self.weight_refit_mode = weight_refit_mode
        self.refit_request = {}
        self.node_manager = NodeManager(initial_nodes=nodes)

        # 选择层分配策略：贪婪或动态规划
        allocator_class = (
            GreedyLayerAllocator if strategy == "greedy" else DynamicProgrammingLayerAllocator
        )
        self.dynamic_pipelines_router = routing_strategy == "dp"
        # TODO: expose DP's alpha
        self.layer_allocator = allocator_class(
            model_info=model_info,
            node_management=self.node_manager,
            dynamic_pipelines_router=self.dynamic_pipelines_router,
            rebalance_threshold=rebalance_threshold,
            water_filling_max_iterations=water_filling_max_iterations,
            trim_layers_on_turning_points=trim_layers_on_turning_points,
        )

        # 确保调度器和分配器共享同一个节点列表，避免状态不一致
        self.nodes = self.layer_allocator.nodes
        self.node_id_to_node: Dict[str, Node] = self.layer_allocator.node_id_to_node
        self.min_nodes_bootstrapping = min_nodes_bootstrapping

        # 选择请求路由策略：动态规划或轮询
        self.request_router = (
            DynamicProgrammingRouting()
            if routing_strategy == "dp"
            else RoundRobinOverFixedPipelinesRouting(self.node_manager)
        )

        self._request_queue: "queue.Queue[RequestSignal]" = queue.Queue()
        self.request_arrival_horizon_sec = request_arrival_horizon_sec
        self.heartbeat_timeout = heartbeat_timeout
        self._arrival_ts: Deque[float] = deque()


        # 用于主循环编排的事件队列（线程安全）
        self._pending_joins: "queue.Queue[Node]" = queue.Queue()
        self._pending_leaves: "queue.Queue[str]" = queue.Queue()
        self._pending_node_updates: "queue.Queue[Tuple[str, Optional[int], Optional[float], Optional[Dict[str, float]], Optional[bool]]]" = (queue.Queue())

        # Concurrency controls
        self._stop_event: threading.Event = threading.Event()
        self._wake_event: threading.Event = threading.Event()
        self._bootstrapped_event: threading.Event = threading.Event()
        self._node_count_cv: threading.Condition = threading.Condition()
        self._event_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._alloc_log_thread: Optional[threading.Thread] = None

        # 线程安全的引导状态
        self._bootstrapped: bool = False
        self._bootstrapped_event: threading.Event = threading.Event()
        logger.debug(
            f"Scheduler initialized, min_nodes_bootstrapping {self.min_nodes_bootstrapping}, "
            f"Layer allocations trategy {strategy}, Request routing strategy {routing_strategy}."
        )
        self._node_assigned_request_count: Dict[str, int] = {}


        # 如果有足够的节点，立即尝试进行初始分配（急切引导）
        try:
            if len(self.nodes) >= self.min_nodes_bootstrapping:
                logger.debug(
                    f"Eager allocation attempt with {len(self.nodes)} nodes (min required: {self.min_nodes_bootstrapping})"
                )
                self.layer_allocator.global_allocation()
        except Exception:  # best-effort eager allocation
            pass


    # 编排辅助方法
    def bootstrap(self, *, clear_existing: bool = False, skip_warmup: bool = False) -> bool:
        """引导过程：
        此方法可用于初始引导和全局重新平衡。
        当 clear_existing=True 时，它会在执行全局分配之前先取消所有现有分配（重新平衡行为）。
        当 clear_existing=False 时，它在现有状态之上执行分配（初始引导行为）。

        Args:
            clear_existing: 如果为 True，则在重新分配前清除所有现有分配。用于全局重新平衡。默认为 False。
            skip_warmup: 如果为 True，则跳过预热和截断步骤。默认为 False。

        Returns:
            如果成功建立完整的流水线，则返回 True；否则返回 False。
        """

        # 仅在初始引导时检查节点数量（重新平衡时不检查）
        if not clear_existing and len(self.nodes) < self.min_nodes_bootstrapping:
            logger.debug(
                f"Bootstrapping deferred: have {len(self.nodes)} nodes; need >= {self.min_nodes_bootstrapping}"
            )
            return False


        # 如果是重新平衡，则清除现有分配
        if clear_existing:
            logger.debug("Performing global rebalance (clearing existing allocations)")
            self._bootstrapped = False
            self._bootstrapped_event.clear()
            overide_min_node_check = True
        else:
            # If we already bootstrapped, return True
            if self._bootstrapped_event.is_set():
                logger.info("[Scheduler] Already bootstrapped, returning Success")
                return True
        # Check if we have enough nodes for bootstraping
        if (
            self.node_manager.num_nodes < self.min_nodes_bootstrapping
            and not overide_min_node_check
        ):
            logger.info(
                f"[Scheduler] Bootstrap deferred: have {self.node_manager.num_nodes} nodes; need >= {self.min_nodes_bootstrapping}"
            )
            return False

  
        # 执行全局分配
        success = self.layer_allocator.global_allocation()
        if not success:
            logger.warning("Global allocation failed to produce a full pipeline")
            # Stay un-bootstrapped so future joins can retry bootstrap.
            self._bootstrapped_event.clear()
            return False

        assignments = self.node_manager.list_node_allocations(self.num_layers)
        logger.info(f"[Scheduler] Post Bootstrap Layer Assignments: {assignments}")


        # 可选的预热步骤，用于查找转折点并截断节点范围
        # 重新平衡场景跳过预热（可以通过 skip_warmup=False 覆盖）
        if not skip_warmup and self.request_warm_up_for_reshard > 0:
            self._run_warmup_and_truncate()
            assignments = self.list_node_allocations()
            logger.debug(f"Layer allocator assignments after turn-point warm-up: {assignments}")

        if not self.layer_allocator.has_full_pipeline():
            logger.warning("Bootstrapping failed to produce a full pipeline")
            return False

        self._bootstrapped = True
        self._bootstrapped_event.set()
        # Snapshot at INFO after bootstrap since allocations/pipelines may have materially changed.
        self.emit_alloc_log_snapshot(reason="Post Bootstrap")
        return True

    def list_node_allocations(self) -> List[Tuple[str, int, int]]:
        """List the allocations of all nodes.
        列出所有节点的分配情况。
        """
        return self.layer_allocator.list_node_allocations()


    # 预热和重新分片
    def _run_warmup_and_truncate(self, override_warmup_count: int = 0) -> None:
        """Run a brief warm-up to detect truncation points and shrink shards.
        运行简短的预热以检测截断点并收缩分片。

        Uses layer-level DP turning points (node_id, layer_idx, kind):
        - kind == "tail": drop [layer_idx, end) on that node
        - kind == "head": drop [start, layer_idx) on that node
        使用层级动态规划转折点（node_id, layer_idx, kind）：
        - kind == "tail": 在该节点上丢弃 [layer_idx, end)
        - kind == "head": 在该节点上丢弃 [start, layer_idx)

        Note: Always uses DynamicProgrammingRouting for finding turning points,
        regardless of the current request_router type, since turning points
        detection requires layer-level DP analysis.
        注意：始终使用 DynamicProgrammingRouting 查找转折点，无论当前的 request_router 类型如何，
        因为转折点检测需要层级 DP 分析。

        Args:
            override_warmup_count: If > 0, use this value instead of request_warm_up_for_reshard.
                Default is 0, which means use request_warm_up_for_reshard.
        """
        nodes_list = list(self.nodes)
        if not nodes_list:
            return
        num_layers = self.model_info.num_layers


        # 预热请求的数量可用于重复检测，但对于我们的 DP 模型，单次通过已足够；我们重复以平滑噪声。
        warmup_count = (
            override_warmup_count if override_warmup_count > 0 else self.request_warm_up_for_reshard
        )

        agg_turns: Dict[Tuple[str, int, str], int] = {}
        for _ in range(warmup_count):
            turns = DynamicProgrammingRouting.find_turning_points(nodes_list, num_layers)
            for t in turns:
                agg_turns[t] = agg_turns.get(t, 0) + 1


        # 对一致观察到的转折点应用截断
        # 注意：必须使用 layer_allocator.allocate/deallocate 以正确更新内部状态（node_allocation 字典和 layer_to_load）
        for node_id, layer_idx, kind in agg_turns:
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if node is None or node.start_layer is None or node.end_layer is None:
                continue
            if min_refit_time is None:
                min_refit_time = cur_node_refit_time
            else:
                min_refit_time = min(min_refit_time, cur_node_refit_time)
        if min_refit_time is not None:
            self.last_refit_time = min_refit_time
        return self.last_refit_time

    def update_node_info(
        self,
        node: Node,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
        is_active: Optional[bool] = None,
        last_refit_time: Optional[float] = 0.0,
    ) -> None:
        """更新节点信息。
        包括：当前请求数、层延迟、与其他节点的 RTT、活跃状态等。
        """
        if current_requests is not None:
            node.current_requests = current_requests
        if layer_latency_ms is not None:
            node.set_layer_latency_ms(layer_latency_ms)
        if new_rtt_to_nodes is not None:
            node.rtt_to_nodes = new_rtt_to_nodes
        if is_active is not None:
            node.is_active = is_active
        if last_refit_time > 0.0:
            node.last_refit_time = last_refit_time
        node.last_heartbeat = time.time()

    # Async-style event enqueuers for main loop
    # 用于主循环的异步风格事件入队器
    def enqueue_join(self, node: Node) -> None:
        """加入事件入队。
        """
        logger.debug(f"Enqueueing join event for node {node.node_id}")
        self._pending_joins.put(node)
        self._wake_event.set()

    def enqueue_leave(self, node_id: str) -> None:
        """离开事件入队。
        """
        self._pending_leaves.put(node_id)
        self._wake_event.set()

    def enqueue_node_update(
        self,
        node_id: str,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
        is_active: Optional[bool] = None,
        last_refit_time: Optional[float] = 0.0,
    ) -> None:
        """节点更新事件入队。
        """
        self._pending_node_updates.put(
            (
                node_id,
                current_requests,
                layer_latency_ms,
                new_rtt_to_nodes,
                is_active,
                last_refit_time,
            )
        )
        self._wake_event.set()

    def checking_node_heartbeat(self) -> None:
        """检查所有节点的心跳。
        如果节点超时，则强制其离开。
        """
        for node in self.nodes:
            if not node.is_active:
                continue
            if time.time() - node.last_heartbeat > self.heartbeat_timeout:
                logger.debug(f"Node {node.node_id} heartbeat timeout")
                # Route leave through the event loop so global rebalance/reboot is serialized.
                self.enqueue_leave(node.node_id)

    # Dynamic node management
    # 动态节点管理
    def join(self, node: Node, bootstrap: bool = False) -> None:
        """添加一个节点到分配中，并刷新计划和已具体化的节点。
        """
        logger.debug(
            "Joining node %s (kv_ratio=%.2f, param_ratio=%.2f, manual_assignment=%s)",
            node.node_id,
            node.kvcache_mem_ratio,
            node.param_mem_ratio,
            node.manual_layer_assignment,
        )
        self.layer_allocator.declare(node)

        # 手动层分配跳过引导等待
        if node.manual_layer_assignment:

            # 手动层分配：使用节点指定的层
            if node.start_layer is None or node.end_layer is None:
                raise ValueError(
                    f"Node {node.node_id} has manual_layer_assignment=True "
                    f"but start_layer ({node.start_layer}) or end_layer ({node.end_layer}) is None"
                )
            logger.info(
                f"Manual layer assignment for node {node.node_id}: "
                f"layers [{node.start_layer}, {node.end_layer})"
            )

            # 直接分配指定的层，无需自动分配
            self.layer_allocator.allocate(node, node.start_layer, node.end_layer)

            # 检查手动分配是否已覆盖完整的流水线
            if self.layer_allocator.has_full_pipeline():
                if not self._bootstrapped:
                    logger.info(
                        "[Scheduler] Manual layer assignments have established a full pipeline; "
                        "marking scheduler as bootstrapped"
                    )
                    self._bootstrapped_event.set()
        elif not bootstrap:
            # 自动层分配（仅在引导后）
            self.layer_allocator.join(node)

        # 如果 bootstrap=True 且非手动，节点仅被声明（分配推迟到 bootstrap()）

        # 通知等待者节点数量已更改
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def leave(self, node_id: str) -> None:
        """从分配中移除一个节点，并刷新计划和已具体化的节点。
        """
        if node_id not in self.layer_allocator.node_id_to_node:
            raise ValueError(f"Node {node_id} not found in nodes")
        node = self.node_id_to_node[node_id]
        logger.debug(
            "Leaving node %s (start=%s, end=%s)", node_id, node.start_layer, node.end_layer
        )
        self.layer_allocator.leave(node_id)
        if self.layer_allocator.should_global_rebalance():
            logger.debug("Global rebalance triggered due to node leave")

            # 统计手动与自动节点
            manual_count = sum(1 for n in self.nodes if n.manual_layer_assignment)
            total_count = len(self.nodes)
            logger.debug(
                f"Node count: {manual_count} manual, {total_count - manual_count} automatic"
            )
            if manual_count == total_count:
                logger.debug("All nodes are manual assignment, skipping global rebalance")
            elif manual_count > 0:
                logger.error(
                    f"Mixed assignment detected ({manual_count} manual, {total_count - manual_count} automatic); skipping rebalance"
                )
            else:
                # 所有节点均为自动，先尝试调整，如有需要再重新平衡
                if not self.layer_allocator.has_full_pipeline():
                    logger.debug(
                        "No full pipeline after node leave, attempting warmup and truncate"
                    )
                    self._run_warmup_and_truncate(override_warmup_count=1)
                    if not self.layer_allocator.has_full_pipeline():
                        self.bootstrap(clear_existing=True, skip_warmup=True)
                    else:
                        logger.debug(
                            "Pipeline recovered through warmup and truncate, skipping global rebalance"
                        )
                else:
                    self.bootstrap(clear_existing=True, skip_warmup=True)

        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def receive_request(self, request: RequestSignal) -> None:
        """将请求添加到等待池中。
        """
        self._request_queue.put(request)
        self._wake_event.set()
        now = time.time()
        self._arrival_ts.append(now)
        logger.debug(
            "Received request %s (queue_size=%d)", request.request_id, self._request_queue.qsize()
        )
        # 修剪旧时间戳以保持到达率窗口有界
        horizon = self.request_arrival_horizon_sec
        while self._arrival_ts and now - self._arrival_ts[0] > horizon:
            self._arrival_ts.popleft()

    def dispatch_next_request(self) -> Optional[Tuple[str, List[str], float]]:
        """路由等待池中的下一个请求；返回 (request_id, path, latency)。
        """
        try:
            req = self._request_queue.get_nowait()
        except queue.Empty:
            req = None
        if req is None:
            return None
        # 使用请求路由器查找最佳路径
        path, latency = self.request_router.find_optimal_path(self.nodes, self.num_layers)
        req.routing_table = path
        # 更新简单的负载计数器
        for node_id in path:
            n = self.node_manager.get(node_id)
            if n is not None:
                self.node_manager.add_request(node_id)
        logger.debug(
            "Dispatched request %s via path %s (est_lat=%.2fms)", req.request_id, path, latency
        )
        return req.request_id, path, latency

    def _format_current_allocations_snapshot(self) -> str:
        assignments = self.node_manager.list_node_allocations(self.num_layers)
        header = f"Current allocations ({len(assignments)} nodes)"
        sep = "-" * len(header)
        lines: List[str] = [header, sep]
        for node_id, start_layer, end_layer in assignments:
            node = self.node_manager.get(node_id)
            if node is None:
                raise ValueError(f"Node {node_id} not found in node manager")
            # Snapshot values to avoid recomputing/logging side-effects twice
            capacity = node.max_requests
            current = node.current_requests
            latency = node.layer_latency_ms
            latency_str = "inf" if latency == float("inf") else f"{latency:.2f}"
            n_hosted_requests = 0
            if node_id in self.node_manager.node_assigned_request_count:
                n_hosted_requests = self.node_manager.node_assigned_request_count[node_id]
            lines.append(
                "  %-16s layers [%3d, %3d) | load %3d/%-3d | latency %7s ms | assigned request count %3d | active %s"
                % (
                    node_id,
                    start_layer,
                    end_layer,
                    current,
                    capacity,
                    latency_str,
                    n_hosted_requests,
                    node.is_active,
                )
            )
        if len(lines) == 2:
            lines.append("  (none)")
        return "\n".join(lines)

    def _format_rr_registered_pipelines_snapshot(self) -> str:
        if self.routing_strategy != "rr":
            return ""
        pipelines = self.node_manager.get_registered_pipelines()
        p_header = f"Registered pipelines ({len(pipelines)})"
        p_sep = "-" * len(p_header)
        lines: List[str] = [p_header, p_sep]
        # Include capacity summary in the RR snapshot message.
        per_pipeline_min, total_capacity, cur_capacity = self.report_pipeline_capacity()
        if per_pipeline_min is None:
            lines.append("Capacity: (no registered pipelines)")
        else:
            lines.append(
                f"Capacity: total={total_capacity} cur={cur_capacity} per_pipeline={per_pipeline_min}"
            )
        if not pipelines:
            lines.append("  (none)")
            return "\n".join(lines)

        for pid in sorted(pipelines.keys()):
            node_ids = pipelines.get(pid, [])
            lines.append("  pipeline %-3d | stages=%d" % (pid, len(node_ids)))
            for idx, nid in enumerate(node_ids):
                n = self.node_manager.get(nid)
                if n is None:
                    lines.append("    [%02d] %-16s (missing)" % (idx, nid))
                    continue
                s = -1 if n.start_layer is None else int(n.start_layer)
                e = -1 if n.end_layer is None else int(n.end_layer)
                lat = n.layer_latency_ms
                lat_str = "inf" if lat == float("inf") else f"{lat:.2f}"
                lines.append(
                    "    [%02d] %-16s layers [%3d, %3d) | load %3d/%-3d | latency %7s ms | active %s"
                    % (
                        idx,
                        nid,
                        s,
                        e,
                        n.current_requests,
                        n.max_requests,
                        lat_str,
                        n.is_active,
                    )
                )
        return "\n".join(lines)

    def emit_alloc_log_snapshot(self, *, reason: Optional[str] = None) -> str:
        """Update `self.alloc_log_snapshot` and emit it.

        - Periodic/heartbeat snapshots (no reason) are logged at DEBUG.
        - Mutating events (join/leave/bootstrap) provide a reason and are logged at INFO.
        """
        try:
            if self.routing_strategy == "rr":
                snapshot = self._format_rr_registered_pipelines_snapshot()
            else:
                snapshot = self._format_current_allocations_snapshot()
        except Exception as exc:
            snapshot = f"(failed to build allocation snapshot: {exc})"
            logger.warning("Allocation snapshot build error: %s", exc)

        self.alloc_log_snapshot = snapshot

        if reason:
            logger.info("Allocation snapshot (%s)\n%s", reason, snapshot)
        else:
            logger.debug("Allocation snapshot\n%s", snapshot)
        return snapshot

    def report_pipeline_capacity(
        self,
    ) -> Tuple[Optional[Dict[int, Tuple[int, int]]], int, int]:
        """Helper to report the current pipeline capacity.

        Returns:
            per_pipeline_min: Dict of pipeline id -> (min_node_capacity, min_remaining_capacity).
            total_capacity: The total capacity of all registered pipelines.
            cur_capacity: The current capacity (counting existing request load) of all registered pipelines.
        """
        return self.node_manager.report_pipeline_capacity()

    def run(self, *, poll_interval: float = 0.05, allocation_log_interval: float = 5.0) -> None:
        """ 并发运行调度器，直到调用 `stop()`。
        启动后台线程进行事件处理（joins/leaves/updates/heartbeats）和请求分发。
        启动时，等待至少 `min_nodes_bootstrapping` 个节点出现，然后运行 `bootstrap()`。
        """
        logger.debug("Running scheduler")
        self._stop_event.clear()

        # 首先启动事件线程，以便在等待引导时处理加入事件
        self._event_thread = threading.Thread(
            target=self._event_loop, args=(poll_interval,), name="SchedulerEventLoop", daemon=True
        )
        self._event_thread.start()

        # 引导门控：等待足够的节点
        if not self._wait_for_bootstrap(poll_interval):
            return

        # 仅在成功引导后启动分发器
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            args=(poll_interval,),
            name="SchedulerDispatcher",
            daemon=True,
        )
        self._dispatch_thread.start()

        # 启动定期分配日志记录线程
        def _alloc_log_loop() -> None:
            """定期记录当前层分配情况。
            """
            while not self._stop_event.is_set():
                try:
                    self.emit_alloc_log_snapshot()
                except Exception as exc:
                    logger.warning(f"Allocation logger error: {exc}")

                # After bootstrap, periodically check if *all* nodes report active and log once.
                if self._bootstrapped_event.is_set():
                    nodes = self.node_manager.nodes
                    if nodes:
                        all_active = all(n.is_active for n in nodes)
                        if all_active and not self._all_nodes_active_logged:
                            logger.info("All %d nodes are active", len(nodes))
                            self._all_nodes_active_logged = True
                        elif not all_active:
                            self._all_nodes_active_logged = False
                time.sleep(max(1.0, allocation_log_interval))

        self._alloc_log_thread = threading.Thread(
            target=_alloc_log_loop, name="SchedulerAllocLogger", daemon=True
        )
        self._alloc_log_thread.start()

        # 阻塞直到请求停止
        try:
            while not self._stop_event.is_set():
                time.sleep(max(0.5, poll_interval))
        finally:
            if self._event_thread is not None:
                self._event_thread.join(timeout=2.0)
            if self._dispatch_thread is not None:
                self._dispatch_thread.join(timeout=2.0)
            if self._alloc_log_thread is not None:
                self._alloc_log_thread.join(timeout=2.0)

    # === 模块化工作循环 ===
    def _event_loop(self, poll_interval: float) -> None:
        """Process joins/leaves/updates and perform heartbeat checks.
        处理加入/离开/更新并执行心跳检查。
        """
        last_hb_check = 0.0
        while not self._stop_event.is_set():
            self._process_node_updates()
            self._process_joins()
            self._process_leaves()
            now = time.time()
            if now - last_hb_check >= max(0.5, poll_interval):
                self.checking_node_heartbeat()
                last_hb_check = now
            self._wake_event.wait(timeout=poll_interval)
            self._wake_event.clear()

    def _dispatch_loop(self, poll_interval: float) -> None:
        """运行时持续分发传入请求。
        """
        while not self._stop_event.is_set():
            try:
                req = self._request_queue.get(timeout=poll_interval)
                if req is None:
                    continue
                path, path_rtt = self.request_router.find_optimal_path(
                    self.node_manager.active_nodes, self.num_layers
                )
                logger.debug(f"Path RTT: {path_rtt}")
                req.routing_table = path
                for node_id in path:
                    self.node_manager.add_request(node_id)
                logger.debug(
                    "Dispatched request %s via path %s", getattr(req, "request_id", "?"), path
                )
            except queue.Empty:
                continue

    def _wait_for_bootstrap(self, poll_interval: float) -> bool:
        """等待直到有足够的节点，然后运行引导。如果停止则返回 False。
        """
        logger.debug("Waiting for bootstrap")
        while not self._stop_event.is_set() and not self._bootstrapped_event.is_set():
            with self._node_count_cv:
                self._node_count_cv.wait(timeout=max(0.5, poll_interval))
        return not self._stop_event.is_set()

    def _process_node_updates(self) -> None:
        """从队列应用挂起的节点统计信息更新。
        """
        while True:
            try:
                node_id, cur, lat, rtts, is_active, last_refit_time = (
                    self._pending_node_updates.get_nowait()
                )
            except queue.Empty:
                break
            node = self.node_manager.get(node_id)
            if node is None:
                logger.warning(f"Node {node_id} not found in node manager, ignore the update")
                continue
            self.update_node_info(
                node,
                current_requests=cur,
                layer_latency_ms=lat,
                new_rtt_to_nodes=rtts,
                is_active=is_active,
                last_refit_time=last_refit_time,
            )

    def _process_joins(self) -> None:
        """处理挂起的加入事件，遵循分配的引导状态。
        """
        joined_any = False
        had_manual_assignment = False
        while True:
            try:
                node = self._pending_joins.get_nowait()
            except queue.Empty:
                break
            # 在引导期间（尚未建立完整流水线），仅声明节点；不进行动态分配。
            # 引导后，允许动态轻量级加入。
            # 例外：手动层分配无论引导状态如何都会立即处理。
            self.join(node, bootstrap=not self._bootstrapped_event.is_set())
            joined_any = True
            if node.manual_layer_assignment:
                had_manual_assignment = True

        # 如果尚未引导（例如，在离开触发的重新平衡之后）且新节点刚刚加入，
        # 当我们有足够的节点时，立即尝试贪婪引导。
        # 如果未能产生完整的流水线，我们将在后续加入时重试。
        # 如果使用了手动分配，则跳过引导（它们在内部处理引导）。
        if joined_any and not self._bootstrapped_event.is_set() and not had_manual_assignment:
            if self.node_manager.num_standby_nodes >= self.min_nodes_bootstrapping:
                try:
                    ok = self.bootstrap()
                    if not ok:
                        logger.debug(
                            "Bootstrap attempt after join did not produce a full pipeline; will retry on future joins"
                        )
                except Exception as exc:
                    logger.debug(
                        f"Bootstrap attempt after join failed: {exc}; will retry on future joins"
                    )
            else:
                logger.debug(
                    "Deferring bootstrap: have %d nodes; need >= %d",
                    self.node_manager.num_standby_nodes,
                    self.min_nodes_bootstrapping,
                )

    def _process_leaves(self) -> None:
        """安全地处理挂起的离开事件。
        """
        while True:
            try:
                node_id = self._pending_leaves.get_nowait()
            except queue.Empty:
                break
            try:
                self.leave(node_id)
                removed_any = True
            except Exception as exc:
                logger.warning(f"Leave failed for {node_id}: {exc}")

        # After draining all leaves, decide whether to do a single global rebalance.
        if not removed_any:
            return

        if not self.layer_allocator.should_global_rebalance():
            return

        nodes = self.node_manager.nodes
        logger.warning("Global rebalance triggered due to node leave")

        # Count manual vs automatic nodes
        manual_count = sum(1 for n in nodes if n.manual_layer_assignment)
        total_count = len(nodes)
        logger.debug(f"Node count: {manual_count} manual, {total_count - manual_count} automatic")
        if total_count == 0:
            logger.debug("No nodes left after leave(s); skipping global rebalance")
            return
        if manual_count == total_count:
            logger.debug("All nodes are manual assignment, skipping global rebalance")
            return
        if manual_count > 0:
            logger.error(
                f"Mixed assignment detected ({manual_count} manual, {total_count - manual_count} automatic); skipping rebalance"
            )
            return

        # Move active nodes to standby and re-bootstrap (reboot) once.
        self.node_manager.standby([n.node_id for n in self.node_manager.active_nodes])
        assert (
            self.node_manager.num_standby_nodes == self.node_manager.num_nodes
        ), "All active nodes should be moved to standby"
        assert self.node_manager.num_active_nodes == 0, "No active nodes before re-bootstrap"
        logger.warning("Re-bootstrapping for global rebalance")
        try:
            self.bootstrap(reboot=True)
        finally:
            # Ensure snapshot reflects post-rebalance state even if bootstrap fails.
            self.emit_alloc_log_snapshot(reason="after global rebalance")

    def stop(self) -> None:
        """向后台线程发送停止信号并唤醒任何等待者。
        """
        self._stop_event.set()
        self._wake_event.set()
        with self._node_count_cv:
            self._node_count_cv.notify_all()

    def need_more_nodes(self):
        return (
            not self._bootstrapped_event.is_set()
            and self.node_manager.num_standby_nodes >= self.min_nodes_bootstrapping
        )
