class EvoAlg():
    def __init__(self):
        pop = []

    def ea_output(carstate):
        steering = 0
        command = []
        if carstate[3] < 0:
            carstate[3] = 1
        if carstate[0] < 0.2 or carstate[3] = 1:
            command.append(1)
        else: command.append(0)
        if carstate[0] > 0.2 and carstate[3] < 1:
            command.append(1)
        else: command.append(0)
        if (carstate[1] < -0.2 or carstate[2] > 0.2) and carstate[4] < 1:
            steering = carstate[4] + 0.1
        if (carstate[1] > 0.2 or carstate[2] < -0.2) and carstate[4] > -1:
            steering = carstate[4] - 0.1
        command.append(steering)
        return command
