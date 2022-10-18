class PipeLine:
    def __init__(self, components, role):
        self.components = components
        self.role = role

    def add_component(self, component):
        self.components.append(component)

    def fit(self, trainset, validset):
        for component in self.components[:-1]:
            trainset = component.fit(trainset, role=self.role)
            validset = component.fit(validset, role=self.role)
        self.components[-1].fit(trainset, validset, role=self.role)

    def score(self, testset):
        for component in self.components[:-1]:
            testset = component.fit(testset, role=self.role)
        self.components[-1].score(testset, role=self.role)
