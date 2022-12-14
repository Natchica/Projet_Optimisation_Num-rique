@doc doc"""
Tester l'algorithme du Pas de Cauchy

# Entrées :
   * afficher : (Bool) affichage ou non des résultats de chaque test

# Les cas de test (dans l'ordre)
   * a = 0 => b = 0 (g = 0) : s = 0
              b != 0 : H = 0
   * a > 0 : delta_lim = norm(g)^3/(transpose(g)*H*g) => 0.9*delta_lim : s = - (delta*g)/norm(g)
                                                         1.1*delta_lim : s = - g*norm(g)^2/(transpose(g)*H*g)
   * a < 0 : H = -I
"""
function tester_pas_de_cauchy(afficher::Bool,Pas_De_Cauchy::Function)
	@testset "Pas de Cauchy" begin
		@testset "Cas test a = 0" begin
            g1 = [1e-15; 1e-15]
            H1 = [10 1;
                  1  2]
            Δ1 = 1.2
		    s1,e1 = Pas_De_Cauchy(g1,H1,Δ1)
			@testset "b = 0 (g = 0)" begin
		    @test iszero(s1)
            @test e1 == 0
		    end
            g2 = [10; 2]
            H2 = [0 0;
                  0 0]
            Δ2 = 2.1
            s2,e2 = Pas_De_Cauchy(g2,H2,Δ2)
            @testset "b != 0 (H = 0)" begin
            @test !iszero(s2)
            @test e2 == 1
            end
		end
        @testset "Cas test a > 0" begin
            g = [12; 5]
            H = [10 1;
                 1  2]
            Δlim = norm(g)^3/(transpose(g)*H*g)
            s3,e3 = Pas_De_Cauchy(g,H,0.9*Δlim)
			@testset "Δ == 0.9Δlim" begin
		    @test isapprox(s3,-(0.9*Δlim)/norm(g)*g,atol=tol_erreur)
            @test e3 == 1
            end
            s4,e4 = Pas_De_Cauchy(g,H,1.1*Δlim)
            @testset "Δ == 1.1Δlim" begin
            @test isapprox(s4,-norm(g)^2/(transpose(g)*H*g)*g,atol=tol_erreur)
            @test e4 == -1
		    end
        end
        @testset "Cas test a < 0" begin
            g = [12; 5]
            H = -I
            Δ = 1.12
            s5,e5 = Pas_De_Cauchy(g,H,Δ)
			@testset "H = -I" begin
            @test isapprox(s5,-Δ/norm(g)*g,atol=tol_erreur)
            @test e5 == 1
            end
        end
	end
end