import numpy as np
import uuid
import scipy.optimize
import obsAvoidance
class beastieSingleAgent:
    
    def __init__(self,
                 id : int = 0,
                 state_dim : int = 2,
                 control_dim : int = 2,
                 n_steps : int = 50,
                 initial_pos : np.ndarray = np.array([2.0, 1.0]),
                 initial_cov : float = 1.0,
                 final_goal : np.ndarray = np.array([0.0, 0.0]),
                 Q : float = 10.0,
                 R :float = 1.0,
                 lambda_cov : float = 2000.0
                 ) -> None:
        """Initialize the single agent system with dynamics and returns the id of the agent"""
        
        self.STATE_DIM = state_dim
        self.CONTROL_DIM = control_dim
        self.n_steps = n_steps
        self.INITIAL_POS = initial_pos
        self.INITIAL_COV = initial_cov
        self.FINAL_GOAL = final_goal
        self.FINAL_COV = 0.1
        self.Q = Q * np.eye(state_dim)
        self.R = R * np.eye(control_dim)
        self.LAMBDA_COV = lambda_cov
        self.id = str(id) +  "__" + str(uuid.uuid4())[:8]
        self.curr_belief = np.hstack((self.INITIAL_POS, self.INITIAL_COV))
        self.curr_pos = self.INITIAL_POS.copy()
        self.curr_cov = self.INITIAL_COV
        self.curr_input = np.zeros(self.CONTROL_DIM)
        self.static_obs = [np.array([2.71, 0.117])]
        self.dynamic_obs = []
        self.ts = 0.1
        self.bel_nom = None
        self.u_nom = None
        print("Created agent with id:", self.id, " at position:", self.INITIAL_POS, " with goal:", self.FINAL_GOAL)
        
        
        
    def dynamics(self, state : np.ndarray, control : np.ndarray) -> np.ndarray:
        """Dynamics of the system """
        return state + self.ts * control
    
    def diff_dynamics(self, state : np.ndarray) -> np.ndarray: 
        """Derivative of the dynamics wrt state"""

        return np.eye(self.STATE_DIM)
    
    def observation_model(self, state: np.ndarray) -> np.ndarray:
        """Observation model"""
        return state + np.random.randn(self.STATE_DIM) * 0.1
    
    def diff_observation_model(self, state : np.ndarray) -> np.ndarray:
        """Derivative of the observation model wrt state"""
        return np.eye(self.STATE_DIM)
    
    def W_noise(self, state : np.ndarray) -> np.ndarray:
        """Process noise covariance"""
        return (0.3 + 0.5 * (5 - state[0])**2) * np.eye(self.STATE_DIM)

    def belief_dynamics(self, b, u):
        """Simplified belief dynamics."""
        mean, cov = b[0:self.STATE_DIM], np.eye(self.STATE_DIM) * b[self.STATE_DIM]
        A = self.diff_dynamics(mean)
        pred_mean = self.dynamics(mean, u) 
        pred_cov = A @ cov @ A.T
        C = np.eye(self.STATE_DIM)  
        W =self.W_noise(mean) 
        cov_new = pred_cov -  pred_cov @ C.T @ np.linalg.inv(C @ pred_cov @ C.T + W) @ C @ pred_cov
        mean_new = pred_mean 
        return np.hstack((mean_new, cov_new.diagonal()[0]))
    
    def cost_function(self, beliefs, controls):
        """Compute total cost for a given belief and control trajectory."""
        # Extract mean deviations from beliefs
        mean_devs = beliefs[:, 0:self.STATE_DIM] - self.FINAL_GOAL
        vars = beliefs[:, self.STATE_DIM]
        
        cost_state  = np.sum(np.einsum('ti,ij,tj->t', mean_devs, self.Q, mean_devs))
        cost_control = np.sum(np.einsum('ti,ij,tj->t', controls, self.R, controls))
        # final_var = beliefs[-1, STATE_DIM]
        cost_var = np.sum(vars**2 ) * self.LAMBDA_COV
        terminalCostMean = mean_devs[-1] @ self.Q*100 @ mean_devs[-1].T
        # cost_terminal = LAMBDA_COV * vars
        total_cost = cost_state + cost_control + cost_var + terminalCostMean 
        return total_cost
        
    def get_initial_guess(self, n_steps=None):
        """Generate an initial guess for the optimization."""
        if n_steps is not None:
            self.n_steps = n_steps
        initial_beliefs = np.zeros((n_steps, self.STATE_DIM + 1))  
        tt = np.linspace(0, 1, n_steps)
        initial_beliefs[:, 0] = tt
        initial_beliefs[:, 1] = tt
        initial_beliefs[:, self.STATE_DIM] = self.INITIAL_COV
        initial_controls = np.zeros((n_steps, self.STATE_DIM))
        
        return initial_beliefs, initial_controls
    
    def receive_positions(self,dict_list : dict | list[dict]):
        """Receive  positions of other agents as dynamic obstacles.
        The dict_list should contain dictionaries with keys `id`, `position`, `velocity`, and `covariance`."""
        
        if isinstance(dict_list, dict):
            dict_list = [dict_list]
            
        ids_presents = [d['id'] for d in self.dynamic_obs]
        
        for d in dict_list:
            
            if d['id'] == self.id:
                continue
            
            if d['id'] in ids_presents:
                idx = ids_presents.index(d['id'])
                self.dynamic_obs[idx] = d
            else:
                self.dynamic_obs.append(d)

    def send_positions(self):
        """Send the current position, velocity, and covariance of this agent."""
        return {'id': self.id, 'position': self.curr_pos.copy(), 'velocity': self.curr_input.copy(), 'covariance': self.curr_belief[self.STATE_DIM]}
    

    
    def optimize_trajectory(self, initial_pos=None, initial_cov=None, final_goal=None, final_cov=None,n_steps=None):
        """Optimize the trajectory using scipy.optimize.minimize."""
        if initial_pos is None:
            initial_pos = self.INITIAL_POS
        if initial_cov is None:
            initial_cov = self.INITIAL_COV
        if final_goal is None:
            final_goal = self.FINAL_GOAL
        if final_cov is None:
            final_cov = self.FINAL_COV
        if n_steps is None:
            n_steps = self.n_steps
        initial_beliefs, initial_controls = self.get_initial_guess(n_steps=n_steps)
        x0 = np.hstack((initial_beliefs.flatten(), initial_controls.flatten()))
                        
 
                
            
        def objective(x):
            beliefs = x[:(n_steps)*(1+self.STATE_DIM)].reshape((n_steps, self.STATE_DIM + 1))
            controls = x[(n_steps)*(1+self.STATE_DIM):].reshape((n_steps, self.STATE_DIM))
            return self.cost_function( beliefs, controls)
        def constraint_ineq(x):
            beliefs = x[:(n_steps)*(1+self.STATE_DIM)].reshape((n_steps, self.STATE_DIM + 1))
            controls = x[(n_steps)*(1+self.STATE_DIM):].reshape((n_steps, self.STATE_DIM))
            cons = []
            dynamic_obstacle_pos = [i["position"] for i in self.dynamic_obs.copy()]
            dynamic_obstacle_vel = [i["velocity"] for i in self.dynamic_obs.copy()]
            dynamic_obstacle_cov = [i["covariance"] for i in self.dynamic_obs.copy()]
            for i in range(n_steps):
                for j,sob in enumerate(self.static_obs):
                    As,bs,*_ = obsAvoidance.obstacle_halfspace_from_gaussian(beliefs[i,:self.STATE_DIM], r_vehicle=0.3, delta=0.7, center=sob, Sigma=np.eye(self.STATE_DIM)*0.5,r_obstacle=0.2)
                    cons.append( - As @ beliefs[i,0:self.STATE_DIM] + bs)
                if i == 0:
                    for j , dob in enumerate(dynamic_obstacle_pos):
                        Ad,bd,*_ = obsAvoidance.obstacle_halfspace_from_gaussian(beliefs[i,:self.STATE_DIM], r_vehicle=0.3, delta=0.7, center=dob, Sigma=np.eye(self.STATE_DIM)*0.5,r_obstacle=dynamic_obstacle_cov[j])
                        dynamic_obstacle_bel = self.belief_dynamics(np.hstack([dob,dynamic_obstacle_cov[j]]), dynamic_obstacle_vel[j])
                        dynamic_obstacle_pos[j] = dynamic_obstacle_bel[0:self.STATE_DIM]
                        dynamic_obstacle_cov[j] = dynamic_obstacle_bel[self.STATE_DIM]
                        cons.append( - Ad @ beliefs[i,0:self.STATE_DIM] + bd)
            return np.hstack(cons)
        def constraint_eq(x):
            beliefs = x[:(n_steps)*(1+self.STATE_DIM)].reshape((n_steps, self.STATE_DIM + 1))
            controls = x[(n_steps)*(1+self.STATE_DIM):].reshape((n_steps, self.STATE_DIM))
            cons = []
            for i in range(n_steps-1):
                cons.append(beliefs[i+1] - self.belief_dynamics(beliefs[i], controls[i]) ) 

            cons.append(beliefs[0, 0:self.STATE_DIM] - initial_pos)
            cons.append(beliefs[0, self.STATE_DIM] - initial_cov)
            cons.append(controls[-1, 0:self.STATE_DIM] )
            # cons.append(beliefs[-1, 0:self.STATE_DIM] - final_goal)
            # cons.append(beliefs[-1, STATE_DIM] - FINAL_COV)
            return np.hstack(cons)
        
        constraints = {'type': 'eq', 'fun': constraint_eq}
        ineq_constraints = {'type': 'ineq', 'fun': constraint_ineq}
        
        # Set bounds only for controls (no bounds for beliefs)
        bounds = [(None, None)] * (n_steps * (self.STATE_DIM + 1)) + [(-2, 2)] * (n_steps * self.STATE_DIM)
        
        result = scipy.optimize.minimize(objective, x0, constraints=[constraints], method='SLSQP', bounds=bounds,options={'maxiter': 10000, 'ftol': 1e-4, 'disp': False})
        
        # if not result.success:
        #     raise RuntimeError("Optimization failed: " + result.message)
        
        print(f"For agent {self.id} Optimization success:", result.success, "Message:", result.message)

        optimized_beliefs = result.x[:(n_steps)*(1+self.STATE_DIM)].reshape((n_steps, 1+ self.STATE_DIM))
        optimized_controls = result.x[(n_steps)*(1+self.STATE_DIM):].reshape((n_steps, self.STATE_DIM))
        
        return optimized_beliefs, optimized_controls
    
    
    
    def EKF(self,b,u) -> np.ndarray:
        mean, cov = b[0:self.STATE_DIM], np.eye(self.STATE_DIM) * b[self.STATE_DIM]
        # Linearize and predict
        A = self.diff_dynamics(mean) # Jacobian of f w.r.t. x is identity
        pred_cov = A @ cov @ A.T
        W = self.W_noise(mean) 
        pred_mean = self.dynamics(mean, u) + pred_cov @ np.linalg.inv(pred_cov + W) @ (self.observation_model(self.dynamics(mean,u)) - self.dynamics(mean,u))

        # EKF update with predicted observation = g(pred_mean) (max likelihood observation assumption)
        C = np.eye(self.STATE_DIM)  # Jacobian of g w.r.t. x is identity
    # Observation noise covariance
        cov_new = pred_cov -  pred_cov @ C.T @ np.linalg.inv(C @ pred_cov @ C.T + W) @ C @ pred_cov
        mean_new = pred_mean  # mean remains at prediction (no correction):contentReference[oaicite:29]{index=29}
        return np.hstack((mean_new, cov_new.diagonal()[0]))

    def BLQR(self,beliefs, controls,timestep,bel_curr,n_steps=None):
        """Compute the Belief LQR solution for a given belief and control trajectory."""
        if n_steps is None:
            n_steps = self.n_steps
        s = np.zeros((n_steps, self.STATE_DIM , self.STATE_DIM ))
        s[n_steps-1] = self.LAMBDA_COV * np.eye(self.STATE_DIM)
        A = self.diff_dynamics(bel_curr)
        B = np.eye(self.STATE_DIM )
        for ti in reversed(range(n_steps-1)):
            s[ti] = self.Q + A.T @ s[ti+1] @ A - A.T @ s[ti+1] @ B @ np.linalg.inv(self.R + B.T @ s[ti+1] @ B) @ B.T @ s[ti+1] @ A
            if timestep == ti:
                c = controls[ti,:]
                b_diff = bel_curr[0:self.STATE_DIM] -  beliefs[ti,0:self.STATE_DIM] 
                u =  c - np.linalg.inv(self.R + B.T @ s[ti] @ B) @ B.T @ s[ti] @ A @ (b_diff)
                if u is None:
                    raise ValueError("Control computation resulted in None")
                return u.clip(-2,2)
        
    def step(self):
        """Take a single step using the BLQR controller."""
        bel_curr = self.curr_belief
        flagReplan = False
        if self.bel_nom is None or self.u_nom is None:
            # steps_remaining = self.n_steps - timestep
            # n_steps =int( steps_remaining + (1/6) * self.n_steps )
            n_steps = self.n_steps
            flagReplan = True
            self.agentTimestep = 0
            # if steps_remaining < 0:
            #     raise ValueError("No steps remaining for optimization.")
            print("###############")
            print("")
            print("Replanning for agent ", self.id, " with n_steps = ", n_steps)
            b_nom, u_nom  = self.optimize_trajectory(initial_pos=self.curr_pos, initial_cov=self.curr_belief[self.STATE_DIM], final_goal=self.FINAL_GOAL, n_steps=n_steps)
            self.bel_nom = b_nom
            self.u_nom = u_nom

        u = self.BLQR(self.bel_nom, self.u_nom, self.agentTimestep, bel_curr)
        new_bel = self.EKF(bel_curr, u)
        self.curr_belief = new_bel
        if flagReplan:
            if np.linalg.norm(self.curr_belief[0:self.STATE_DIM] - self.bel_nom[1,0:self.STATE_DIM]) > 0.1:
                self.bel_nom = None
                self.u_nom = None
                print("##################")
                print("")
                print("Replanning for agent ", self.id)
            else:
                print("No replanning needed for agent ", self.id)
        else:
            if np.linalg.norm(self.curr_belief[0:self.STATE_DIM] - self.bel_nom[self.agentTimestep + 1,0:self.STATE_DIM]) > 0.1:
                self.bel_nom = None
                self.u_nom = None
                print("##################")
                print("")
                print("Replanning for agent ", self.id)
            else:
                print("##################")
                print("")
                print("No replanning needed for agent ", self.id)
        self.agentTimestep += 1
        return new_bel, u
    
    def plan(self):
        
        final_traj = []
        final_traj.append(self.curr_belief.copy())
        while np.linalg.norm(self.curr_pos - self.FINAL_GOAL) > 0.1:
            optimized_beliefs, optimized_controls = self.optimize_trajectory(initial_pos=self.curr_pos, initial_cov=self.curr_belief[self.STATE_DIM], final_goal=self.FINAL_GOAL)

            for t in range(self.n_steps - 1):
                u = self.BLQR(optimized_beliefs, optimized_controls,t,self.curr_belief[0:self.STATE_DIM])
                self.curr_belief = self.EKF(self.curr_belief, u)
                self.curr_pos = self.curr_belief[0:self.STATE_DIM]
                final_traj.append(self.curr_belief.copy())
                print(f"Time step {t}: Control: {u}, New Belief: {self.curr_belief}")
                if np.linalg.norm(self.curr_belief[0:self.STATE_DIM] - optimized_beliefs[t+1,0:self.STATE_DIM]) > 0.1:
                    print("Replanning!")
                    break 
                if np.linalg.norm(self.curr_pos - self.FINAL_GOAL) < 0.1:
                    print("Reached the goal!")
                    break
        
        print("Planning completed.")
        print(f"Final Position: {self.curr_pos}, Final Belief: {self.curr_belief}")
        final_traj = np.array(final_traj)
        return final_traj
            
        
# if __name__ == "__main__":
#     agent = beastieSingleAgent()
#     final_traj = agent.plan()
    
#     import matplotlib.pyplot as plt
#     plt.figure()
#     plt.axis('equal')
#     plt.plot(final_traj[:,0], final_traj[:,1], '-o', label='Planned Path')
#     plt.plot(agent.INITIAL_POS[0], agent.INITIAL_POS[1], 'go', label='Start')
#     plt.plot(agent.FINAL_GOAL[0], agent.FINAL_GOAL[1], 'ro', label='Goal')
#     for sob in agent.static_obs:
#         circle = plt.Circle(sob, 0.2, color='r')
#         plt.gca().add_artist(circle)
#     for k,dob in enumerate(agent.dynamic_obs):
#             circle = plt.Circle(dob, 0.2, color='orange',alpha= 0.1)
#             plt.gca().add_artist(circle)
#             for i in range(len(final_traj)):
#                 dob = agent.belief_dynamics(np.hstack((dob["position"],0.5)),dob["covariance"])[0:agent.STATE_DIM]
#                 circle = plt.Circle(dob, 0.2, color='orange', alpha=0.1)
#                 plt.gca().add_artist(circle)

#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Agent Path with Obstacle')
#     plt.legend()
#     plt.grid()
#     plt.show()  