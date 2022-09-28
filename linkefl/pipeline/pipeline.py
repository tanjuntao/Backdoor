from linkefl.pipeline.component import Component


class PipeLine:
    """记录调用的函数以及每个函数对应的参数和输入输出，可生成参数表"""

    # Use pipeline to inject pool, messenger and logger?

    def __init__(self):
        self.components = list()

    def add_component(self, component: Component):
        self.components.append(component)

    def run(self):
        for component in self.components:
            component.run()
