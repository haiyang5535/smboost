import json

def main():
    hard_tasks = []
    with open('benchmarks/data/test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            task = json.loads(line)
            # Find the 'Hard' tasks
            if str(task.get('difficulty')).lower() == 'hard':
                # Convert to spec schema: {task_id, prompt, test_code, entry_point, difficulty, source}
                
                task_id = task.get('question_id', '')
                prompt = task.get('question_content', '')
                # Combine public and private tests if available
                public_tests = task.get('public_test_cases', []) or []
                private_tests = task.get('private_test_cases', []) or []
                if isinstance(public_tests, str):
                    try:
                        public_tests = json.loads(public_tests)
                    except:
                        public_tests = [public_tests]
                if isinstance(private_tests, str):
                    try:
                        private_tests = json.loads(private_tests)
                    except:
                        private_tests = [private_tests]
                        
                if not isinstance(public_tests, list): public_tests = [public_tests]
                if not isinstance(private_tests, list): private_tests = [private_tests]
                
                test_cases = public_tests + private_tests
                test_code = json.dumps(test_cases)
                
                entry_point = task.get('starter_code', '')
                source = task.get('platform', 'livecodebench')
                
                hard_tasks.append({
                    'task_id': task_id,
                    'prompt': prompt,
                    'test_code': test_code,
                    'entry_point': entry_point,
                    'difficulty': 'Hard',
                    'source': source
                })
                
            if len(hard_tasks) == 50:
                break
                
    with open('benchmarks/data/livecodebench_hard_v1.jsonl', 'w', encoding='utf-8') as f:
        for t in hard_tasks:
            f.write(json.dumps(t) + '\n')
            
    print(f"Saved {len(hard_tasks)} hard tasks to benchmarks/data/livecodebench_hard_v1.jsonl")

if __name__ == '__main__':
    main()