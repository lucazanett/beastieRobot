from singleAgent import beastieSingleAgent
import random
import numpy as np

class multiAgentSim:
    def __init__(self, num_agents: int = 1) -> None:
        self.agents :  list[beastieSingleAgent] = []
        for i in range(num_agents):
            if i != 0:
                initial_pos = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
                while True:
                    if np.linalg.norm(initial_pos - prev_initial_pos) < 2.0:
                        initial_pos += np.array([random.uniform(-2, 2), random.uniform(-2, 2)])
                    else:
                        break
            else:   
                initial_pos = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
            prev_initial_pos = initial_pos.copy()
            agent = beastieSingleAgent(id = i, initial_pos=initial_pos)
            self.agents.append(agent)
            
            
    def setDynamicObstacles(self):
        dict_list = []
        for agent in self.agents:
            dict_list.append(agent.send_positions())
        for agent in self.agents:
            agent.receive_positions(dict_list)
            
            
    
    def simulate(self):
        timestep =0
        self.setDynamicObstacles()
        trajectories = {agent.id: [agent.curr_pos] for agent in self.agents}
        timestep = 0        
        while True:
            sum = 0

            for agent in self.agents:
                if np.linalg.norm(agent.curr_pos - agent.FINAL_GOAL) < 0.1:
                    sum += 1
                    print(f"Agent {agent.id} has reached its goal.")
                    continue
                agent_bel , _ = agent.step(timestep)
                trajectories[agent.id].append(agent_bel[0:agent.STATE_DIM])
            self.setDynamicObstacles()
            timestep += 1
            if sum == len(self.agents):
                print("All agents have reached their goals.")
                break
            if timestep > 100:
                print("Max timesteps reached.")
                break
            
        return trajectories
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import csv, queue, sys, time
import os

# ----- worker process: owns a persistent Agent instance -----
def agent_loop(agent, cmd_q: mp.Queue, res_q: mp.Queue):
    """
    Runs in a child process. Keeps 'agent' alive across iterations.
    Expects dict messages: {"type": "step", "t": int, "world": optional}
    """
    import numpy as _np  # local import for child
    while True:
        msg = cmd_q.get()
        if msg is None or msg.get("type") == "stop":
            break

        if msg["type"] == "step":
            t = msg["t"]
            world = msg.get("world")
            # If your agent needs world updates, apply them here:
            if world is not None and hasattr(agent, "apply_world_update"):
                agent.apply_world_update(world)

            # Advance one step; this mutates the agent INSIDE THIS PROCESS
            agent_bel, extra = agent.step()

            # Build a small, picklable result
            state = _np.asarray(agent_bel[: agent.STATE_DIM], float)
            pos = state[:2]
            goal = _np.asarray(agent.FINAL_GOAL[:2], float)
            reached = bool(_np.linalg.norm(pos - goal) < 0.1)

            res_q.put({
                "id": agent.id,
                "state": state,     # length = STATE_DIM
                "pos": pos,         # x,y convenience
                "reached": reached,
            })

# ----- main script -----
def main():
    sim = multiAgentSim(num_agents=4)
    sim.setDynamicObstacles()
    timestep = 0

    # Start one persistent worker per agent (agent instance lives in that process)
    ctx = mp.get_context("spawn")  # GUI-safe
    workers = {}                   # id -> dict(proc, cmd_q, res_q)
    for a in sim.agents:
        cmd_q = ctx.Queue()
        res_q = ctx.Queue()
        p = ctx.Process(target=agent_loop, args=(a, cmd_q, res_q), daemon=True)
        p.start()
        workers[a.id] = {"proc": p, "cmd_q": cmd_q, "res_q": res_q, "goal": np.asarray(a.FINAL_GOAL[:2], float)}

    # Trajectory history for plotting
    trajectories = {a.id: [np.asarray(a.curr_pos, float)] for a in sim.agents}

    # ---- plotting setup (single persistent figure) ----
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Agent trajectories (live)")
    ax.set_aspect("equal", adjustable="box")

    lines = {}
    for a in sim.agents:
        p0 = np.asarray(a.curr_pos, float)
        (line,) = ax.plot([p0[0]], [p0[1]], marker="o", linewidth=1.5, label=f"A{a.id}")
        lines[a.id] = line
        g = np.asarray(a.FINAL_GOAL, float)
        ax.scatter([g[0]], [g[1]], marker="x", label=f"G{a.id}")

    # initial view box
    all_pts = np.vstack([np.asarray(a.curr_pos, float) for a in sim.agents] +
                        [np.asarray(a.FINAL_GOAL, float) for a in sim.agents])
    pad = 0.5
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend(loc="upper right", ncols=2)
    plt.show(block=False)

    try:
        while True:
            # If you have world updates to broadcast, compute them here.
            # Example: world = sim.getDynamicObstaclesState()
            world = None

            # Tell each worker to step
            for aid, w in workers.items():
                w["cmd_q"].put({"type": "step", "t": timestep, "world": world})

            # Collect one result per worker, keeping TkAgg responsive
            pending = set(workers.keys())
            reached_count = 0
            while pending:
                for aid in list(pending):
                    try:
                        msg = workers[aid]["res_q"].get_nowait()
                    except queue.Empty:
                        continue
                    # Update local history with returned state (this is authoritative)
                    s = np.asarray(msg["state"], float)
                    trajectories[aid].append(s)

                    # CSV dump (x,y)
                    with open(f"trajectories_agent_{aid}.csv", "w", newline="") as f:
                        wcsv = csv.writer(f)
                        wcsv.writerow(["x", "y"])
                        for pt in trajectories[aid]:
                            wcsv.writerow(pt[:2])

                    if msg["reached"]:
                        reached_count += 1

                    pending.remove(aid)

                # keep GUI alive during waits
                fig.canvas.draw_idle()
                plt.pause(0.005)

            # environment update for next tick (if needed)
            sim.setDynamicObstacles()
            timestep += 1

            # ---- update the figure (reuse artists) ----
            for aid, line in lines.items():
                pts = np.asarray(trajectories[aid])
                if pts.size:
                    line.set_data(pts[:, 0], pts[:, 1])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            plt.pause(0.001)

            if reached_count == len(workers):
                print("All agents have reached their goals.")
                break
            if timestep > 100:
                print("Max timesteps reached.")
    finally:
        # Graceful shutdown
        for w in workers.values():
            w["cmd_q"].put({"type": "stop"})
        for w in workers.values():
            w["proc"].join(timeout=2.0)

        plt.ioff()
        plt.show()
if __name__ == "__main__":
    for fname in os.listdir("."):
        if fname.startswith("trajectories") and fname.endswith(".csv"):
            try:
                os.remove(fname)
            except Exception as e:
                print(f"Could not remove {fname}: {e}")
    mp.set_start_method("spawn", force=True)  # safer with GUIs on macOS/Linux
    main()