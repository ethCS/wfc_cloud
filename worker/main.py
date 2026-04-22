import base64
import json
import os
import io
import concurrent.futures
from flask import Flask, request
from google.cloud import storage, firestore
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit, prange
from PIL import Image

app = Flask(__name__)

# Cloud Clients
storage_client = storage.Client()
firestore_client = firestore.Client(database="wfc-db")

# ------------------------------------------------------------------------
# 1. PATTERN EXTRACTION (Vectorized + Numba)
# ------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _compute_rules(patterns):
    """Compute the 4-direction adjacency rule table for a set of unique patterns.

    Only directions 0 (up) and 1 (right) are compared directly; 2 (down) and 3 (left)
    are filled by symmetry, halving the O(P^2) comparison work.
    """
    P = patterns.shape[0]
    N = patterns.shape[1]
    rules = np.zeros((P, 4, P), dtype=np.bool_)
    for i in prange(P):
        for j in range(P):
            # dir 0: p1[1:, :] == p2[:-1, :]
            match0 = True
            for a in range(N - 1):
                for b in range(N):
                    for c in range(3):
                        if patterns[i, a + 1, b, c] != patterns[j, a, b, c]:
                            match0 = False
                            break
                    if not match0:
                        break
                if not match0:
                    break
            rules[i, 0, j] = match0

            # dir 1: p1[:, 1:] == p2[:, :-1]
            match1 = True
            for a in range(N):
                for b in range(N - 1):
                    for c in range(3):
                        if patterns[i, a, b + 1, c] != patterns[j, a, b, c]:
                            match1 = False
                            break
                    if not match1:
                        break
                if not match1:
                    break
            rules[i, 1, j] = match1

    for i in prange(P):
        for j in range(P):
            rules[i, 2, j] = rules[j, 0, i]
            rules[i, 3, j] = rules[j, 1, i]
    return rules


def extract_patterns_and_rules(image_array, N=3):
    H, W = image_array.shape[:2]
    padded = np.pad(image_array, ((0, N - 1), (0, N - 1), (0, 0)), mode='wrap')
    windows = sliding_window_view(padded, (N, N, 3))[:H, :W, 0]
    flat = np.ascontiguousarray(windows).reshape(H * W, N * N * 3)
    unique_flat, counts = np.unique(flat, axis=0, return_counts=True)
    unique_patterns = np.ascontiguousarray(unique_flat.reshape(-1, N, N, 3))
    weights = counts.astype(np.float64) / counts.sum()
    rules = _compute_rules(unique_patterns)
    return unique_patterns, weights, rules

# ------------------------------------------------------------------------
# 2. WAVE FUNCTION COLLAPSE (Numba JIT-Compiled)
# ------------------------------------------------------------------------
# Uses per-cell compatibility counts so each propagation event is O(P) rather
# than O(P^2), and Shannon entropy H = log(Σw) - Σ(w·log w)/Σw for cell
# selection (correctly prioritizes the most-constrained cell).
@njit(cache=True)
def execute_wfc(grid_size, num_patterns, rules, weights):
    wave = np.ones((grid_size, grid_size, num_patterns), dtype=np.bool_)

    dy = np.array([-1, 0, 1, 0], dtype=np.int32)
    dx = np.array([0, 1, 0, -1], dtype=np.int32)
    opp = np.array([2, 3, 0, 1], dtype=np.int32)

    # support[y, x, t, d] = number of patterns still live at the direction-d
    # neighbor of (y, x) that support pattern t being here. When this hits 0,
    # pattern t must be eliminated at (y, x).
    # Initial value: count of j with rules[j, opp(d), t] — how many neighbors
    # at direction d of (y, x) are currently allowed to sit next to t.
    init_support = np.zeros((num_patterns, 4), dtype=np.int32)
    for t in range(num_patterns):
        for d in range(4):
            od = opp[d]
            s = 0
            for j in range(num_patterns):
                if rules[j, od, t]:
                    s += 1
            init_support[t, d] = s

    support = np.empty((grid_size, grid_size, num_patterns, 4), dtype=np.int32)
    for y in range(grid_size):
        for x in range(grid_size):
            for t in range(num_patterns):
                for d in range(4):
                    support[y, x, t, d] = init_support[t, d]

    # Precompute weight log terms for Shannon entropy.
    w_log_w = np.zeros(num_patterns, dtype=np.float64)
    for t in range(num_patterns):
        if weights[t] > 0.0:
            w_log_w[t] = weights[t] * np.log(weights[t])

    # Propagation queue. Each (y, x, t) is enqueued at most once over the run.
    max_q = grid_size * grid_size * num_patterns
    q_y = np.empty(max_q, dtype=np.int32)
    q_x = np.empty(max_q, dtype=np.int32)
    q_t = np.empty(max_q, dtype=np.int32)
    q_head = 0
    q_tail = 0

    while True:
        # --- Find minimum Shannon entropy cell ---
        min_entropy = 1e18
        min_y, min_x = -1, -1

        for y in range(grid_size):
            for x in range(grid_size):
                valid_states = 0
                sum_w = 0.0
                sum_wlw = 0.0
                for t in range(num_patterns):
                    if wave[y, x, t]:
                        valid_states += 1
                        sum_w += weights[t]
                        sum_wlw += w_log_w[t]
                if valid_states == 0:
                    return np.zeros((1, 1), dtype=np.int32)
                elif valid_states > 1:
                    entropy = np.log(sum_w) - sum_wlw / sum_w
                    entropy -= np.random.rand() * 1e-6
                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_y = y
                        min_x = x

        if min_y == -1:
            break

        # --- Observe: weighted random pick among valid patterns ---
        total = 0.0
        for t in range(num_patterns):
            if wave[min_y, min_x, t]:
                total += weights[t]
        r = np.random.rand() * total
        acc = 0.0
        chosen = -1
        for t in range(num_patterns):
            if wave[min_y, min_x, t]:
                acc += weights[t]
                if acc >= r:
                    chosen = t
                    break
        if chosen == -1:
            for t in range(num_patterns):
                if wave[min_y, min_x, t]:
                    chosen = t

        # Collapse: eliminate every other pattern, enqueue each removal.
        for t in range(num_patterns):
            if wave[min_y, min_x, t] and t != chosen:
                wave[min_y, min_x, t] = False
                q_y[q_tail] = min_y
                q_x[q_tail] = min_x
                q_t[q_tail] = t
                q_tail += 1

        # --- Propagate ---
        while q_head < q_tail:
            jy = q_y[q_head]
            jx = q_x[q_head]
            j = q_t[q_head]
            q_head += 1

            for d in range(4):
                ny = jy + dy[d]
                nx = jx + dx[d]
                if 0 <= ny < grid_size and 0 <= nx < grid_size:
                    od = opp[d]
                    for t in range(num_patterns):
                        if wave[ny, nx, t] and rules[j, d, t]:
                            support[ny, nx, t, od] -= 1
                            if support[ny, nx, t, od] == 0:
                                wave[ny, nx, t] = False
                                q_y[q_tail] = ny
                                q_x[q_tail] = nx
                                q_t[q_tail] = t
                                q_tail += 1

    # Extract result (first live pattern per cell).
    result = np.zeros((grid_size, grid_size), dtype=np.int32)
    for y in range(grid_size):
        for x in range(grid_size):
            for t in range(num_patterns):
                if wave[y, x, t]:
                    result[y, x] = t
                    break
    return result


def solve_wfc_with_retries(grid_size, num_patterns, rules, weights):
    """Runs the Numba solver in a loop until it succeeds without contradictions."""
    attempts = 0
    while True:
        attempts += 1
        print(f"Executing WFC (Attempt {attempts})...")
        result_grid = execute_wfc(grid_size, num_patterns, rules, weights)
        if result_grid.shape != (1, 1):
            return result_grid

# ------------------------------------------------------------------------
# 3. THE FLASK WORKER / PIPELINE GLUE
# ------------------------------------------------------------------------
@app.route('/', methods=['POST'])
def pubsub_push():
    envelope = request.get_json()
    if not envelope or 'message' not in envelope:
        return 'Bad Request', 400

    msg_data = base64.b64decode(envelope['message']['data']).decode('utf-8')
    work_order = json.loads(msg_data)

    input_bucket = work_order.get('input_bucket')
    input_file = work_order.get('input_filename')
    output_bucket = work_order.get('output_bucket')
    job_id = work_order.get('job_id')
    patch_size = int(work_order.get('patch_size', 3))
    grid_size = int(work_order.get('output_size', 128))

    if not all([input_bucket, input_file, output_bucket, job_id]):
        return 'Missing required info in payload', 400
    
    print(f"Processing Job: {job_id} | Output: {grid_size}x{grid_size} | Patch: {patch_size}")

    try:
        # 1. Download Seed Image
        in_blob = storage_client.bucket(input_bucket).blob(input_file)
        img_bytes = in_blob.download_as_bytes()
        seed_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        seed_array = np.array(seed_img)
        
        # 2. Learn the Rules
        patterns, weights, rules = extract_patterns_and_rules(seed_array, N=patch_size)
        num_patterns = len(patterns)
        
        # 3. Run the Solver with a 5-Minute Circuit Breaker
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(solve_wfc_with_retries, grid_size, num_patterns, rules, weights)
            try:
                result_grid = future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                print(f"Job {job_id} hit the 5-minute limit and was aborted.")
                firestore_client.collection("wfc_jobs").document(job_id).update({"status": "TIMED OUT"})
                return 'Timeout', 200  # Return 200 so Pub/Sub drops the message
                
        # 4. Render the Final Image
        top_left_colors = patterns[:, 0, 0, :]
        final_array = top_left_colors[result_grid]
        final_img = Image.fromarray(final_array, 'RGB')
        out_io = io.BytesIO()
        final_img.save(out_io, format='PNG')
        
        # 5. Upload Results
        out_name = f"generated-{job_id}.png"
        out_blob = storage_client.bucket(output_bucket).blob(out_name)
        out_blob.upload_from_string(out_io.getvalue(), content_type='image/png')
        
        public_url = f"https://storage.googleapis.com/{output_bucket}/{out_name}"

        # 6. Update the "Whiteboard"
        firestore_client.collection("wfc_jobs").document(job_id).update({
            "status": "COMPLETE",
            "output_url": public_url
        })
        
        print(f"Job {job_id} complete. Saved to {public_url}")
        return 'Success', 200

    except Exception as e:
        print(f"Pipeline Error for Job {job_id}: {e}")
        firestore_client.collection("wfc_jobs").document(job_id).update({"status": "ERROR"})
        return 'Internal Server Error', 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)