class MyDescription:
    def __init__(self):
        pass
    def __get__(self,instance,owner):
        return 2

    def __set__(self, instance, value):
        pass

class Main:
    attr = MyDescription()


def testcase1():
    m = Main()
    m.attr = 1
    # m.__dict__['attr'] = 1
    # 输出的是 2
    print(m.attr)

def testcase2():
    m = Main()
    Main.attr = 1
    print(m.attr)

if __name__ == '__main__':
    testcase2()
