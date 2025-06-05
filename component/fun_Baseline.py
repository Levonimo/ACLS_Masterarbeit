import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """
    Schätzt die Baseline einer 1D‐Kurve y mittels Asymmetric Least Squares (Eilers).
    
    Parameter:
    ----------
    y : np.ndarray, Form (T,)
        Die gemessenen Intensitäten eines einzelnen m/z‐Kanals über T Scans.
    lam : float
        Glattheitsparameter (je größer, desto glatter).
        Typische Werte für Chromatographie liegen bei 1e5 ... 1e8.
    p : float
        Asymmetrie‐Parameter (Gewichtung des Faktors).
        p ≈ 0.001 ... 0.1; je kleiner p, desto stärker betont man negative Residuen (Peaks).
    niter : int
        Anzahl der Iterationen zur Aktualisierung der Gewichte w.
    
    Rückgabe:
    --------
    baseline : np.ndarray, Form (T,)
        Die geschätzte Baseline.
    """
    T = y.shape[0]
    # Unterschiedsmatrix 2. Ordnung (T×T), dünnbesetzt
    # D ist (T-2)×T‐Matrix, so dass D @ b = (b[i-1] - 2 b[i] + b[i+1]) für i=1..T-2
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T))
    # C = λ * (D^T @ D) – dünnbesetzte (T×T)-Matrix
    C = lam * (D.T @ D)
    
    # Initialisierung: Gewichte w_i = 1
    w = np.ones(T)
    
    for _ in range(niter):
        # Diagonalmatrix W aus Gewichten
        W = sparse.diags(w, 0)
        # LGS: (W + C) b = W y
        A = W + C
        B = w * y  # elementweise Multiplikation
        # Sparse‐Solve
        baseline = spsolve(A, B)
        
        # Residuen
        resid = y - baseline
        # Aktualisiere Gewichte: je nachdem, ob resid > 0 oder resid < 0
        # wenn resid > 0 (Messwert über Baseline → wahrscheinlicher Peak), dann w_i = p
        # sonst w_i = 1-p
        w = p * (resid > 0) + (1 - p) * (resid < 0)
        # Note: resid==0 → w bleibt 1-p (kann man so lassen)
    
    return baseline
