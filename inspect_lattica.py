
import inspect
from lattica import Lattica

print("--- Lattica Class Attributes ---")
for name in dir(Lattica):
    print(name)

print("\n--- Lattica Instance Attributes ---")
try:
    # Try to build a minimal instance
    l = Lattica.builder().with_listen_addrs(["/ip4/0.0.0.0/tcp/0"]).build()
    for name in dir(l):
        print(name)
except Exception as e:
    print(f"Could not build instance: {e}")
