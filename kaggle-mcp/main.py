import time

print("Starting dummy 5-minute run...")

start = time.time()
duration = 300  # 5 minutes in seconds
step = 0

while time.time() - start < duration:
    elapsed = time.time() - start
    remaining = duration - elapsed
    loss = 1.0 / (step + 1) + 0.01
    step += 1

    if step % 10 == 0 or elapsed < 2:
        print(
            f"Step {step} | "
            f"elapsed: {elapsed:.1f}s | "
            f"remaining: {remaining:.1f}s | "
            f"loss: {loss:.6f}"
        )

    time.sleep(0.5)

total = time.time() - start
print(f"\nDone! Completed {step} steps in {total:.1f}s.")
