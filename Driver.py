import PursuitEvasionEnviornment
import PursuitEvasionSimulator

if name == "__main__":
    enviornment = Enviornment.enviornment()
    enviornment.reset()
    Steps = 45
    for i in range(Steps):
        enviornment.step()
    States.enviornment.states.state_history