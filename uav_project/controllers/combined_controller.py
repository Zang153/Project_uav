from uav_project.controllers.cascade_controller import CascadeController
from uav_project.controllers.delta_controller import DeltaController

class CombinedController:
    """
    Combines the UAV CascadeController and the DeltaController.
    """
    def __init__(self, uav_model):
        self.uav_controller = CascadeController(uav_model)
        self.delta_controller = DeltaController(uav_model)
        self.uav = uav_model

    def update(self, sim_time):
        """
        Updates both controllers.
        """
        self.uav_controller.update(sim_time)
        self.delta_controller.update(sim_time)

    def set_target_position(self, pos):
        """
        Sets target for UAV controller.
        Delta controller has its own internal trajectory generator.
        """
        self.uav_controller.set_target_position(pos)

    def get_log_data(self):
        """
        Returns log data. Currently delegates to UAV controller.
        """
        # TODO: Extend Logger to log Delta states if needed.
        return self.uav_controller.get_log_data()

    def print_state(self):
        """
        Prints the state of the system.
        """
        self.uav_controller.print_state()
