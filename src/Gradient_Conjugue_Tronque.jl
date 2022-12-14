@doc doc"""
#### Objet
Cette fonction calcule une solution approchée du problème

```math
\min_{||s||< \Delta}  q(s) = s^{t} g + \frac{1}{2} s^{t}Hs
```

par l'algorithme du gradient conjugué tronqué

#### Syntaxe
```julia
s = Gradient_Conjugue_Tronque(g,H,option)
```

#### Entrées :   
   - g : (Array{Float,1}) un vecteur de ``\mathbb{R}^n``
   - H : (Array{Float,2}) une matrice symétrique de ``\mathbb{R}^{n\times n}``
   - options          : (Array{Float,1})
      - delta    : le rayon de la région de confiance
      - max_iter : le nombre maximal d'iterations
      - tol      : la tolérance pour la condition d'arrêt sur le gradient

#### Sorties:
   - s : (Array{Float,1}) le pas s qui approche la solution du problème : ``min_{||s||< \Delta} q(s)``

#### Exemple d'appel:
```julia
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
xk = [1; 0]
options = []
s = Gradient_Conjugue_Tronque(gradf(xk),hessf(xk),options)
```
"""
function Gradient_Conjugue_Tronque(g,H,options)

    "# Si option est vide on initialise les 3 paramètres par défaut"
    if options == []
        delta = 2
        max_iter = 100
        tol = 1e-6
    else
        delta = options[1]
        max_iter = options[2]
        tol = options[3]
    end

    n = length(g)
    Tol_abs = sqrt(eps())
    Tol_rel = 1e-15

    j = 0
    g₀ = g
    s₀ = zeros(n)
    p₀ = -g

    gⱼ = g₀
    sⱼ = s₀
    pⱼ = p₀

    q(s) = transpose(g)*s + (transpose(s)*H*s)/2

    while j < 2*n && norm(gⱼ) > max(norm(g₀)*Tol_rel,Tol_abs)
        κⱼ = transpose(pⱼ)*H*pⱼ
        if κⱼ <= 0
            # la racine de ||sⱼ + σpⱼ|| = ∆ pour laquelle q(sⱼ + σpⱼ) est la plus petite
            σ₁,σ₂ = racineNe∆(sⱼ,pⱼ,delta)
            if q(sⱼ+σ₁*pⱼ) > q(sⱼ+σ₂*pⱼ)
                σⱼ = σ₂
            else
                σⱼ = σ₁
            end 
            return sⱼ + σⱼ*pⱼ
        end
        αⱼ = transpose(gⱼ)*gⱼ/κⱼ
        if norm(sⱼ + αⱼ*pⱼ) >= delta
            # la racine positive de ||sⱼ + σpⱼ|| = ∆
            σⱼ,_ = racineNe∆(sⱼ,pⱼ,delta)
            return sⱼ + σⱼ*pⱼ
        end
        sⱼ₊₁ = sⱼ + αⱼ*pⱼ
        gⱼ₊₁ = gⱼ + αⱼ*H*pⱼ
        βⱼ = (transpose(gⱼ₊₁)*gⱼ₊₁)/(transpose(gⱼ)*gⱼ)
        pⱼ₊₁ = -gⱼ₊₁ + βⱼ*pⱼ

        sⱼ = sⱼ₊₁
        pⱼ = pⱼ₊₁
        gⱼ = gⱼ₊₁
        j += 1

    end

    return sⱼ
end

function racineNe∆(s,p,Δ)
    a = norm(p)^2
    b = 2*transpose(s)*p
    c = norm(s)^2 - Δ^2
    D = b^2 - 4*a*c
    σ₁ = (-b+sqrt(D))/(2*a)
    σ₂ = (-b-sqrt(D))/(2*a)
    return σ₁,σ₂
end
