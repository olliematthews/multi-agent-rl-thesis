from multiprocessing import Process
from time import sleep


class Boi:
    def run(self):
        for i in range(3):
            print(i)
            sleep(1)

        print("DONE")


b = Boi()

b.run()

print("ok")
Process(b.run).start()
