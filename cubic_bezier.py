import torch
import config as cfg

def cubic_bezier(p, ts):
    return (1-ts)**3*p[0] + 3*(1-ts)**2*ts*p[1]+3*(1-ts)*ts**2*p[2]+ts**3*p[3]

def cubic_bezier_arc_3D(px, py, pz, unit_arc=0.1):
    t = 0
    ts = torch.tensor([0]).to(cfg.device)
    while t <= 1:
        dx = 3*(1 - t)**2 * (px[1] - px[0]) + 6 * (1-t) * t * (px[2] - px[1]) + 3*t**2*(px[3] - px[2])
        dy = 3*(1 - t)**2 * (py[1] - py[0]) + 6 * (1-t) * t * (py[2] - py[1]) + 3*t**2*(py[3] - py[2])
        dz = 3*(1 - t)**2 * (pz[1] - pz[0]) + 6 * (1-t) * t * (pz[2] - pz[1]) + 3*t**2*(pz[3] - pz[2])

        dt = unit_arc / torch.sqrt(dx**2 + dy**2 + dz**2)

        t = t + dt

        ts = torch.hstack((ts, t))
    ts = ts[:-1]
    xs = cubic_bezier(px, ts)
    ys = cubic_bezier(py, ts)
    zs = cubic_bezier(pz, ts)

    return torch.vstack((xs, ys, zs))