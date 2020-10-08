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
    ('Baseline', 'news', 'ml', 'txt', 10),
    ('LR', 'news', 'ml', 'txt', 10),
    ('LR-S', 'news', 'ml', 'txt', 10),
    ('LGBM', 'news', 'ml', 'txt', 10),
    ('LR-H', 'news', 'ml', 'txt', 1),
    ('LR-SH', 'news', 'ml', 'txt', 1),
]

for model, data, problem, dtype, n_iter in tests:
    logger.info(f'training: {model} {data}')
    exit_code = subprocess.call([
        "D:\\Users\\Ritvik\\Anaconda3\\envs\\ailab\\python.exe",
        os.path.join(os.getcwd(), "train.py"), 
        f"-a {model}", 
        f"-d {data}",
        f"-p {problem}",
        f"-t {dtype}",
        f"-n {n_iter}"
    ])

    subprocess.call([
        "D:\\Users\\Ritvik\\Anaconda3\\envs\\ailab\\python.exe",
        os.path.join(os.getcwd(), "performance_report.py")
    ])
    if exit_code == 0:
        logger.info(f'trained: {model} {data}\n')
    else:
        logger.error(f'failed: {model} {data}\n')
