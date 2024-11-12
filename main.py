from metaflow import FlowSpec, step


class ImmigrationEffectFlow(FlowSpec):

    @step
    def start(self):
        pass
        self.next(self.create_dataset)

    @step
    def create_dataset(self):
        from src.data_processing import OutcomesDataPipeline

        data_pipeline = OutcomesDataPipeline(CA=False)
        self.data = data_pipeline.run()

        self.next(self.fit_instrument_1, self.fit_instrument_2)

    @step
    def fit_instrument_1(self):
        pass
        self.next(self.test_instrument_1)

    @step
    def fit_instrument_2(self):
        pass
        self.next(self.test_instrument_2)

    @step
    def test_instrument_1(self):
        pass
        self.next(self.select_instrument)

    @step
    def test_instrument_2(self):
        pass
        self.next(self.select_instrument)

    @step
    def select_instrument(self, inputs):
        pass
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ImmigrationEffectFlow()
