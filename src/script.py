import subprocess, os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')

file_handler = logging.FileHandler('../training.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

tests = [
    ('LR', '1', 'mc', 'txt'),
    ('LR-S', '1', 'mc', 'txt'),
    ('LGBM', '1', 'mc', 'txt'),
    ('RF', '1', 'mc', 'txt'),
    ('LR', '2', 'ml', 'txt'),
    ('LR-S2', '2', 'ml', 'txt'),
    ('LR-S-A', '2', 'ml', 'txt'),
    ('LR-S-B', '2', 'ml', 'txt'),
    ('LGBM', '2', 'ml', 'txt'),
    ('RF', '2', 'ml', 'txt'),
]

for model, data, problem, dtype in tests:
    logger.info(f'training: {model} {data}')
    exit_code = subprocess.call([
        "D:\\Users\\Ritvik\\Anaconda3\\envs\\ailab\\python.exe",
        os.path.join(os.getcwd(), "train.py"), 
        f"-a {model}", 
        f"-d {data}",
        f"-p {problem}",
        f"-t {dtype}"
    ])

    subprocess.call([
        "D:\\Users\\Ritvik\\Anaconda3\\envs\\ailab\\python.exe",
        os.path.join(os.getcwd(), "performance_report.py")
    ])
    if exit_code == 0:
        logger.info(f'trained: {model} {data}\n')
    else:
        logger.error(f'failed: {model} {data}\n')
