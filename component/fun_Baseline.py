import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye, diags
from scipy.linalg import cho_factor, cho_solve

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


def oals_baseline(y, lam=None, p=None, alpha_factor=5.0, epsilon=1e-5, maxiter=50):
    """
    Optimized Asymmetric Least Squares (O-ALS) zur Baseline-Schätzung für 1D-Kurve y.
    
    Parameter:
    ----------
    y : np.ndarray, Form (T,)
        Gemessene Intensitäten (z. B. ein m/z-Kanal über T Scans).
    lam : float oder None
        Falls None, wird lambda automatisch aus dem Rauschlevel geschätzt.
    p : float oder None
        Asymmetrie-Parameter. Falls None, wird p initial als 0.01 angenommen und ggf. angepasst.
    alpha_factor : float
        Faktor zur Berechnung des Sigmoid-Steilheitsparameters alpha.
    epsilon : float
        Konvergenzschwelle für relative Änderung der Baseline.
    maxiter : int
        Maximale Anzahl Iterationen, falls epsilon nicht vorher erreicht wird.

    Rückgabe:
    --------
    baseline : np.ndarray, Form (T,)
        Geschätzte Baseline-Kurve b(t).
    """
    T = y.shape[0]
    # 1) Rauschschätzung: Wir nehmen z.B. letzten 10% der Punkte (höchste RTs),
    #    dort ist meist nur Bleed + Rauschen, kaum Peaks
    tail_start = int(0.9 * T)
    noise_segment = y[tail_start:]
    sigma_noise = np.std(noise_segment)
    
    # 2) Init: lam automatisch, falls nicht übergeben
    if lam is None:
        # Formel aus Dong & Xu (2024): (max-min / sigma_noise) ^ 2.5
        dynamic_range = np.max(y) - np.min(y)
        lam = (dynamic_range / max(sigma_noise, 1e-8)) ** 2.5
    
    # 3) Init: p automatisch, falls nicht übergeben
    if p is None:
        p = 0.01
    
    # 4) Vorbereitung der dünnbesetzten Matrix C = lam * (D^T D)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T))
    C = lam * (D.T @ D)
    
    # 5) Initial-Baseline per Standard-ALS (eine Iteration mit p fix)
    w = np.ones(T)
    W = sparse.diags(w, 0)
    A = W + C
    B = w * y
    baseline = spsolve(A, B)
    
    # 6) Dynamisches Iterieren mit Soft-Weight-Update
    prev_baseline = baseline.copy()
    for k in range(maxiter):
        # 6.1) Residuen berechnen
        resid = y - baseline
        
        # 6.2) Sigmoid-Steigung alpha
        alpha = alpha_factor / max(sigma_noise, 1e-8)
        
        # 6.3) Weiche Gewichte via Sigmoid-Formel
        #      w_t = p + (1 - 2p) * sigmoid(alpha * resid_t)
        sigmoid_arg = alpha * resid
        # Numerische Stabilität: clamp sigmoid_arg auf [-50, 50]
        sigmoid_arg = np.clip(sigmoid_arg, -50, 50)
        s = 1.0 / (1.0 + np.exp(-sigmoid_arg))
        w = p + (1 - 2*p) * s
        
        # 6.4) Matrice W aktualisieren, Baseline neu berechnen
        W = sparse.diags(w, 0)
        A = W + C
        B = w * y
        baseline = spsolve(A, B)
        
        # 6.5) Relatives Konvergenzkriterium prüfen
        delta = np.linalg.norm(baseline - prev_baseline) / max(np.linalg.norm(prev_baseline), 1e-8)
        if delta < epsilon:
            # Konvergenz erreicht
            break
        prev_baseline = baseline.copy()
        
        # 6.6) Optional: p dynamisch leicht anpassen (falls zu wenige/zu viele Punkte > 2*sigma_noise)
        #           Wir prüfen, ob viele Residuen knapp positiv sind
        over_threshold = np.sum(resid > 2*sigma_noise)
        frac_over = over_threshold / T
        if frac_over < 0.05:
            # Wenige Peaks => Peaks werden mit zu kleinem p nicht deutlich abgehoben
            p = min(p * 1.5, 0.2)
        elif frac_over > 0.30:
            # Viele angebliche Peaks => p ist zu groß, Baseline folgt zu eng
            p = max(p * 0.7, 0.005)
        # Hinweis: Diese Anpassung erfolgt nur in Iterationen, nicht mehrfach pro Iteration
        
    return baseline





def arPLS(y, lambda_, ratio):
    N = len(y)
    D = np.diff(np.eye(N), 2)
    H = lambda_ * D.T @ D
    w = np.ones(N)

    while True:
        W = diags(w, 0, shape=(N, N)).toarray()
        C, lower = cho_factor(W + H)
        z = cho_solve((C, lower), w * y)
        d = y - z

        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)

        wt = 1. / (1. + np.exp(2. * (d - (2. * s - m)) / s))

        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break

        w = wt

    return z
