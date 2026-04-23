import json, base64, zlib, pickle
with open("benchmarks/data/livecodebench_hard_v1.jsonl", "r") as f:
    for line in f:
        task = json.loads(line)
        tc_list = json.loads(task.get("test_code", "[]"))
        loader_count = len([x for x in tc_list if isinstance(x, dict)])
        decoded = []
        for c in tc_list:
            if isinstance(c, str):
                try:
                    c_dec = pickle.loads(zlib.decompress(base64.b64decode(c)))
                    if isinstance(c_dec, list): decoded.extend(c_dec)
                    else: decoded.append(c_dec)
                except Exception: pass
            else:
                decoded.append(c)
        if loader_count != len(decoded):
            print(f"Task {task.get('task_id', '?')}: loader={loader_count}, real={len(decoded)}")
