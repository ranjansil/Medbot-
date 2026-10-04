print("=" * 50, flush=True)
print("CONTAINER STARTED - Loading Python...", flush=True)
print("=" * 50, flush=True)

try:
    print("Importing modules...", flush=True)
    from app import app
    import os
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Flask server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
except Exception as e:
    print(f"FATAL ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    import time
    time.sleep(3600)