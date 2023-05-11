import hydra
import logging

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from joblib import Memory
from tqdm import tqdm

from scipy.stats.qmc import Halton
from aerofoils.aerofoils import NACA
from aerofoils.solver import Foil


logger = logging.getLogger(__name__)


def setup_aesthetic(width, height, aspect, dpi, transparent, use_tex, context,
                    style, palette):

    figsize = width, height

    rc = {
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "font.serif": ["Palatino", "Times", "Computer Modern"],
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{nicefrac}",
        "savefig.dpi": dpi,
        "savefig.transparent": transparent,
    }
    sns.set(context=context, style=style, palette=palette, font="serif", rc=rc)
    return figsize


@hydra.main(version_base=None, config_path="conf", config_name="config")
def plot(cfg: DictConfig) -> None:

    original_path = Path(get_original_cwd())
    # memory = Memory(original_path.joinpath('cachedir'))
    memory = Memory('cachedir')

    output_path = Path.cwd()

    width, height = setup_aesthetic(cfg.plotting.width, cfg.plotting.get('height', cfg.plotting.width / cfg.plotting.aspect), cfg.plotting.aspect, cfg.plotting.dpi, cfg.plotting.transparent, cfg.plotting.use_tex, cfg.plotting.context, cfg.plotting.style, cfg.plotting.palette)
    suffix = f"{width*cfg.plotting.dpi:.0f}x{height*cfg.plotting.dpi:.0f}"
    logger.info(f"original path: {original_path}, output path: {output_path}, suffix: {suffix}")

    if Path('result.csv').exists():
        logger.info("Already processed, skipping...")
        return

    # frames = []
    # for t in np.linspace(0.05, 0.25, 3):

    # for m in np.linspace(0.1, .11, 8):
    # for p in tqdm(np.linspace(.2, .9, 64)):
    # for m in tqdm(np.linspace(.005, .095, 64)):
    # return

    @memory.cache
    def compute(m, p, t):

        naca = NACA(m, p, t, finite_trailing_edge=cfg.foil.training_edge_finite)
        zc, (xl, yl), (xu, yu) = naca(xt)

        x = np.hstack([xu[::-1], xl])
        y = np.hstack([yu[::-1], yl])

        foil = Foil(x, y)

        with foil(mach=cfg.solver.mach, reynolds=cfg.solver.reynolds, 
                  normalize=cfg.solver.normalize, max_iter=cfg.solver.max_iter,
                  max_retries=cfg.solver.max_retries, timeout=cfg.solver.timeout) as foo:

            new_coords = foo.repanel(cfg.foil.trg_resolution)

            # fig, ax = plt.subplots()

            # ax.plot(xt, zc, alpha=0.8, linewidth=1.0, linestyle="dashed", label="mean camber line")

            # ax.scatter(x, y, s=1.2**2, alpha=0.8, label="original pane endpoints")
            # ax.scatter(*new_coords.T, s=1.2**2, alpha=0.8, label="pane endpoints")

            # ax.axvline(p, color="tab:gray", alpha=0.8, linewidth=1.0)

            # ax.set_xlim(-0.1, +1.1)
            # ax.set_ylim(-.3, +.3)

            # ax.legend()

            # plt.tight_layout()

            # for ext in cfg.plotting.extensions:
            #     path = output_path.joinpath(f"profile_{m*100:.2f}_{p*10:.2f}_{t*100:.2f}_{cfg.plotting.context}_{suffix}.{ext}")
            #     fig.savefig(path)
            #     logger.info(f"File saved to '{path}' successfully!")

            # plt.clf()

            frame, metadata, failures = foo.calculate(slice(cfg.alpha_start, cfg.alpha_end, cfg.alpha_step))

        logger.info(foil.coords.shape)
        logger.info(new_coords.shape)

        return frame, metadata, failures

    # m_start, m_end = .005, .095
    # p_start, p_end = .05, .95

    # sampler = Halton(d=2, seed=cfg.seed)

    # U = sampler.random(cfg.n_samples)
    # X = np.empty(shape=(cfg.n_samples, 2))  # len(combo), dim))

    # X[:, 0] = m_start + (m_end - m_start) * U[:, 0]
    # X[:, 1] = p_start + (p_end - p_start) * U[:, 1]

    logger.info(f"Mach: {cfg.solver.mach}, "
                f"Re: {cfg.solver.reynolds}, "
                f"normalize: {cfg.solver.normalize}, "
                f"max. iterations: {cfg.solver.max_iter}, "
                f"max. retries: {cfg.solver.max_retries}, "
                f"timeout: {cfg.solver.timeout}")

    if cfg.foil.half_cosine_spacing:
        beta = np.linspace(0., np.pi, cfg.foil.src_resolution // 2)
        xt = .5 * (1. - np.cos(beta))
    else:
        xt = np.linspace(0., 1., cfg.foil.src_resolution // 2)

    # print(np.mgrid[0.02:0.08:4j, 0.1:0.9:32j, 0.1:0.25:4j].shape)
    # print(np.stack(np.mgrid[0.02:0.08:4j, 0.1:0.9:32j, 0.1:0.25:4j], axis=0).shape)

    # X = np.mgrid[0.02:0.09:16j, 0.2:0.9:32j, 0.1:0.25:4j].reshape(3, -1).T
    # logger.info(X.shape)

    # frames = []
    # for (m, p, t) in tqdm(X):

    logger.info(f"m={cfg.foil.m:.2f}; p={cfg.foil.p:.2f}; t={cfg.foil.t:.2f}")

    # frame, metadata, failures = compute(m, p, t)
    frame, metadata, failures = compute(cfg.foil.m, cfg.foil.p, cfg.foil.t)

    logger.info(f"Failures: {failures}")
    logger.info(f"Results:\n{frame}")
    logger.info(f"Metadata:\n{metadata}")

    frame.to_parquet('result.parquet.gzip', compression='gzip')
    frame.to_csv('result.csv')

    # frames.append(frame.assign(m=m, p=p, t=t))

    # data = pd.concat(frames, axis="index")
    # data.to_parquet('result.parquet.gzip', compression='gzip')

    # data = data.assign(name=data.apply(lambda row: f"{row.m*100:.2f} {row.p*10:.2f} {row.t*100:.2f}", axis="columns"),
    #                    lift_drag_ratio=lambda row: row.CL / row.CD) \
    #     .rename(columns=dict(x=r'$x$', y=r'$y$', 
    #             lift_drag_ratio="lift-to-drag ratio", 
    #             m='maximum camber', p='max camber distance', t='thickness'))

    # logger.info(f"DataFrame:\n{data}")

    # fig, ax = plt.subplots()

    # sns.scatterplot(x='alpha', y='lift-to-drag ratio',
    #                 hue='name',  # palette='crest',
    #                 # units='name', estimator=None,
    #                 # sort=False, 
    #                 data=data, ax=ax)

    # plt.tight_layout()

    # for ext in cfg.plotting.extensions:
    #     path = output_path.joinpath(f"foo_{m:.6f}_{p:.6f}_{cfg.plotting.context}_{suffix}.{ext}")
    #     fig.savefig(path)
    #     logger.info(f"File saved to '{path}' successfully!")

    # plt.clf()

    # fig, ax = plt.subplots()

    # sns.scatterplot(x='CD', y='CL',
    #                 hue='alpha', palette='crest',
    #                 # units='name', estimator=None,
    #                 # sort=False, 
    #                 data=data, ax=ax)

    # plt.tight_layout()

    # for ext in cfg.plotting.extensions:
    #     path = output_path.joinpath(f"test_{m:.6f}_{p:.6f}_{cfg.plotting.context}_{suffix}.{ext}")
    #     fig.savefig(path)
    #     logger.info(f"File saved to '{path}' successfully!")

    # plt.clf()

    # coords = np.hstack([x, y]).T

    # X = np.vstack([
    #     np.vstack([xu, yu]).T[::-1],  # reverse sequence
    #     np.vstack([xl, yl]).T
    # ])

    # logger.info(f"calculating for m={m:.8f}, p={p:.8f}")

    # result = xfoil.from_array(foil.coords, alpha=0.5)

    # print(result)

    # try:
    #     result = xfoil.from_array(X, alpha=0.)
    #     print(result)

    #     frames.append(pd.DataFrame(dict(m=m, p=p, t=t, x=xl, y=yl, kind='lower', **result)))
    #     frames.append(pd.DataFrame(dict(m=m, p=p, t=t, x=xu, y=yu, kind='upper', **result)))
    # except RuntimeError:
    #     logger.error("Failed!")
    #     continue

    # data = pd.concat(frames, axis="index", sort=True) \
    #     .assign(lift_drag_ratio=lambda row: row.CL / row.CD)
    # #     .rename(columns=dict(x=r'$x$', y=r'$y$', lift_drag_ratio="lift-to-drag ratio", m='maximum camber', p='max camber distance', t='thickness'))

    # logger.info(data)


if __name__ == "__main__":
    plot()
