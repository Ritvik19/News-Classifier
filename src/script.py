import subprocess, os

tests = [
    ('DC', '1', 'mc', 'txt'),
    ('LR', '1', 'mc', 'txt'),
    ('DC', '2', 'ml', 'txt'),
    ('LR', '2', 'ml', 'txt'),
    ('LGBM', '2', 'ml', 'txt'),
    ('XGB', '2', 'ml', 'txt'),
    ('RF', '2', 'ml', 'txt'),
    ('LR-G', '2', 'ml', 'txt')
]

for model, data, problem, dtype in tests:
    exit_code = subprocess.call([
        "D:\\Users\\Ritvik\\Anaconda3\\envs\\ailab\\python.exe",
        os.path.join(os.getcwd(), "train.py"), 
        f"-a {model}", 
        f"-d {data}",
        f"-p {problem}",
        f"-t {dtype}"
    ])
    if exit_code == 0:
        print('\ntrained', model, '\n\n')
    else:
        print('\nfailed', model, '\n\n')
