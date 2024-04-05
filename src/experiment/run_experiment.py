from multiprocessing import Manager, Pipe, Pool, Process

from .coordinator import coordinator
from .simulator import simulator


def run_experiment(seed_params):
    """
    Run the actor critic algorithm with a U-function and with the coach frame-
    work. The coordinator is effectively the coach, the simulator is a worker.
    """
    n_workers = 2

    simulator_pool = Pool(processes=n_workers)
    pool_params = []
    pipes = []
    m = Manager()
    queue = m.Queue()
    lock = m.Lock()
    best_workers = m.list()
    best_params = m.list()
    param_history = m.list()
    pipe_conn_1, pipe_conn_2 = Pipe()
    worker_pipe = [pipe_conn_1, pipe_conn_2]
    for worker, worker_pipe_conn in enumerate(worker_pipe):
        dic = seed_params.copy()
        parent_conn, child_conn = Pipe()
        pipes.append(parent_conn)

        dic.update(
            {
                "worker_number": worker,
                "pipe": child_conn,
                "worker_pipe": worker_pipe_conn,
                "queue": queue,
                "lock": lock,
                "param_history": param_history,
            }
        )
        pool_params.append(dic)

    coordinator_args = {
        "pipes": pipes,
        "queue": queue,
        "n_workers": n_workers,
        "n_episodes": seed_params["Sim"]["n_episodes"],
        "hyper_params": seed_params["Hyp"],
        "state_size": 13,
    }

    coordinator_process = Process(
        target=coordinator, args=(coordinator_args, best_workers, best_params)
    )
    coordinator_process.start()
    output = simulator_pool.map(simulator, pool_params)
    coordinator_process.join()
    return [output, list(best_workers), list(best_params), list(param_history)]
