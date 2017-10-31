"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def selectThreshold(p, Ycv):

    e = 0
    f1 = 0

    numIter = 1000
    np.seterr(invalid='ignore')

    for es in np.linspace(np.min(p), np.max(p), numIter):

        tp = sum((Ycv == 1) * (p < es))

        fp = sum((Ycv == 0) * (p < es))

        fn = sum((Ycv == 1) * (p >= es))

        prec = tp / (tp + fp)

        rec = tp / (tp + fn)

        f1s = 2 * prec * rec / (prec + rec)

        if f1s > f1:
            f1 = f1s
            e = es

    return e, f1
