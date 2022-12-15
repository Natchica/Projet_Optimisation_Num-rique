@doc doc"""
#### Objet

Résolution des problèmes de minimisation avec une contrainte d'égalité scalaire par l'algorithme du lagrangien augmenté.

#### Syntaxe
```julia
xmin,fxmin,flag,iter,muks,lambdaks = Lagrangien_Augmente(algo,f,gradf,hessf,c,gradc,hessc,x0,options)
```

#### Entrées
  - algo : (String) l'algorithme sans contraintes à utiliser:
    - "newton"  : pour l'algorithme de Newton
    - "cauchy"  : pour le pas de Cauchy
    - "gct"     : pour le gradient conjugué tronqué
  - f : (Function) la fonction à minimiser
  - gradf       : (Function) le gradient de la fonction
  - hessf       : (Function) la hessienne de la fonction
  - c     : (Function) la contrainte [x est dans le domaine des contraintes ssi ``c(x)=0``]
  - gradc : (Function) le gradient de la contrainte
  - hessc : (Function) la hessienne de la contrainte
  - x0 : (Array{Float,1}) la première composante du point de départ du Lagrangien
  - options : (Array{Float,1})
    1. epsilon     : utilisé dans les critères d'arrêt
    2. tol         : la tolérance utilisée dans les critères d'arrêt
    3. itermax     : nombre maximal d'itération dans la boucle principale
    4. lambda0     : la deuxième composante du point de départ du Lagrangien
    5. mu0, tho    : valeurs initiales des variables de l'algorithme

#### Sorties
- xmin : (Array{Float,1}) une approximation de la solution du problème avec contraintes
- fxmin : (Float) ``f(x_{min})``
- flag : (Integer) indicateur du déroulement de l'algorithme
   - 0    : convergence
   - 1    : nombre maximal d'itération atteint
   - (-1) : une erreur s'est produite
- niters : (Integer) nombre d'itérations réalisées
- muks : (Array{Float64,1}) tableau des valeurs prises par mu_k au cours de l'exécution
- lambdaks : (Array{Float64,1}) tableau des valeurs prises par lambda_k au cours de l'exécution

#### Exemple d'appel
```julia
using LinearAlgebra
algo = "gct" # ou newton|gct
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
c(x) =  (x[1]^2) + (x[2]^2) -1.5
gradc(x) = [2*x[1] ;2*x[2]]
hessc(x) = [2 0;0 2]
x0 = [1; 0]
options = []
xmin,fxmin,flag,iter,muks,lambdaks = Lagrangien_Augmente(algo,f,gradf,hessf,c,gradc,hessc,x0,options)
```

#### Tolérances des algorithmes appelés

Pour les tolérances définies dans les algorithmes appelés (Newton et régions de confiance), prendre les tolérances par défaut définies dans ces algorithmes.

"""
function Lagrangien_Augmente(algo,fonc::Function,contrainte::Function,gradfonc::Function,
        hessfonc::Function,grad_contrainte::Function,hess_contrainte::Function,x0,options)

  if options == []
		epsilon = 1e-2
		tol = 1e-5
		itermax = 1000
		lambda0 = 2
		mu0 = 100
		tho = 2
	else
		epsilon = options[1]
		tol = options[2]
		itermax = options[3]
		lambda0 = options[4]
		mu0 = options[5]
		τ = options[6]
	end

  n = length(x0)
  xmin = x0
  fxmin = 0
  flag = -1
  iter = 0
  μ₀ = mu0
  muks = [mu0]
  λₖ = lambda0
  lambdaks = [lambda0]

  β = 0.9
  η = 0.1258925
  α = 0.1
  ε₀ = 1/μ₀
  ηₖ = η/μ₀^α

  μₖ = μ₀
  εₖ = ε₀

  Lₐ(x) = fonc(x) + transpose(λₖ)*contrainte(x) + μₖ/2*norm(contrainte(x))^2
  gradLₐ(x) = gradfonc(x) + transpose(λₖ)*grad_contrainte(x) + μₖ/2*(2*contrainte(x)*grad_contrainte(x))
  hessLₐ(x) = hessfonc(x) + transpose(λₖ)*hess_contrainte(x) + μₖ/2*(2*(grad_contrainte(x)*transpose(grad_contrainte(x)) + contrainte(x)*hess_contrainte(x)))

  gradL(x) = gradfonc(x) + transpose(λₖ)*grad_contrainte(x)

  while flag == -1
    # un minimiseur du problème sans contrainte min(Lagrangien(x,λₖ,μₖ))
    # où Lagrangien(x,λₖ,μₖ) = f(x) + transpose(μₖ)*c(x) + (μₖ*norm(c(x))^2)/2
    # avec xₖ comme point de départ, en terminant lorsque norm(∇ₓ*Lagrangien(·,λₖ,μₖ)) ≤ εₖ
    if algo == "newton"
      xmin, _, _, _ = Algorithme_De_Newton(Lₐ,gradLₐ,hessLₐ,xmin,[itermax,εₖ,0,0])
    else
      xmin, _, _, _ = Regions_De_Confiance(algo,Lₐ,gradLₐ,hessLₐ,xmin,[10,0.5,2.00,0.25,0.75,2,itermax,εₖ,0,0])
    end

    if norm(contrainte(xmin)) ≤ ηₖ
      λₖ₊₁ = λₖ + μₖ*contrainte(xmin)
      μₖ₊₁ = μₖ
      εₖ₊₁ = εₖ/μₖ
      ηₖ₊₁ = ηₖ/μₖ^β
    else
      λₖ₊₁ = λₖ
      μₖ₊₁ = τ*μₖ
      εₖ₊₁ = ε₀/μₖ₊₁
      ηₖ₊₁ = η/μₖ₊₁^α
    end

    if norm(gradL(xmin)) ≤ tol && norm(contrainte(xmin)) ≤ tol
      flag = 0
    elseif iter + 1 ≥ itermax
      flag = 1
    end 

    push!(muks,μₖ)
    push!(lambdaks,λₖ)

    λₖ = λₖ₊₁
    μₖ = μₖ₊₁
    εₖ = εₖ₊₁
    ηₖ = ηₖ₊₁

    iter += 1

  end

  fxmin = fonc(xmin)

  return xmin,fxmin,flag,iter, muks, lambdaks
end
