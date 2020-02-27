from multiprocessing import Process, Queue
import time


def get_page():
    while (True):
        print("Hello")
        time.sleep(2)

def getData():
    Q.put("This is Data")

def start_get_page(timeout):
    p = Process(target=getData)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()

if __name__ == '__main__':
    Q = Queue()
    timeout = 5
    start_get_page(timeout)
    print(Q.get())
