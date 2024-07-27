problist = LinRange(0.0, 0.5, 10)

SingleSiteMeasurementVariance = load_object("8GateTrajectoryAverageETOneSiteWideEntropyVariance.jld2")
TwoSiteMeasurementVariance = load_object("810CorrectAncillaSetupETTwoSiteWideEntropyVarianceHighTol.jld2")
ThreeSiteMeasurementVariance = load_object("EntagnlementThreeSiteWideVariance.jld2")

CSSingleSiteMeasurementVariance = load_object("8GateTrajectoryAverageCSOneSiteWideEntropyVariance.jld2")
CSTwoSiteMeasurementVariance = load_object("810CorrectAncillaSetupCSTwoSiteWideEntropyVariance.jld2")
L8CSThreeSiteMeasurementVariance = load_object("ChargeSharpeningThreeSiteL=10Variance.jld2")
L10CSThreeSiteMeasurementVariance = load_object("ChargeSharpeningThreeSiteL=10Variance.jld2")

plot(problist, SingleSiteMeasurementVariance[1], label="Single Site ET", linestyle=:dash)
plot!(problist, TwoSiteMeasurementVariance[1], label="Two Site ET", linestyle=:dash)
plot!(problist, ThreeSiteMeasurementVariance[2], label="Three Site ET", linestyle=:dash)

plot!(problist, CSSingleSiteMeasurementVariance[1], label="Single Site CS")
plot!(problist, CSTwoSiteMeasurementVariance[1], label="Two Site CS")
plot!(problist, L8CSThreeSiteMeasurementVariance[1], label="Three Site CS")

scatter!(problist, SingleSiteMeasurementVariance[2], label="")
scatter!(problist, TwoSiteMeasurementVariance[2], label="" )
scatter!(problist, ThreeSiteMeasurementVariance[2], label="" )

scatter!(problist, CSSingleSiteMeasurementVariance[2], label="")
scatter!(problist, CSTwoSiteMeasurementVariance[2], label="" )
scatter!(problist, L8CSThreeSiteMeasurementVariance[1], label="" )

xlabel!("p")
ylabel!("Variance in Entropy")
title!("CS vs ET Entropy Varaince L = 8")
png("CS vs ET Entropy Variance  L = 8, 1000 trials, wide")





plot(problist, SingleSiteMeasurementVariance[2], label="Single Site ET", linestyle=:dash)
plot!(problist, TwoSiteMeasurementVariance[2], label="Two Site ET", linestyle=:dash)
plot!(problist, ThreeSiteMeasurementVariance[2], label="Three Site ET", linestyle=:dash)

plot!(problist, CSSingleSiteMeasurementVariance[2], label="Single Site CS")
plot!(problist, CSTwoSiteMeasurementVariance[2], label="Two Site CS")
plot!(problist, L10CSThreeSiteMeasurementVariance[1], label="Three Site CS")

xlabel!("p")
ylabel!("Variance in Entropy")
title!("CS vs ET Entropy Varaince L = 10")
png("CS vs ET Entropy Variance  L = 10, 1000 trials, Corrected System")


ETSingleSiteMeasurementEntropy = load_object("EntagnlementSingleSiteWideEntropies.jld2")
ETTwoSiteMeasurementEntropy = load_object("EntagnlementTwoSiteWideEntropies.jld2")
ETThreeSiteMeasurementEntropy = load_object("EntagnlementThreeSiteWideEntropies.jld2")

CSSingleSiteMeasurementEntropy = load_object("ChargeSharpeningSingleSiteWideEntropies.jld2")
CSTwoSiteMeasurementEntropy = load_object("ChargeSharpeningOriginalSetupEntropies.jld2")
CSL8ThreeSiteMeasurementEntropy = load_object("ChargeSharpeningThreeSiteL=8Entropies.jld2")
CSL10ThreeSiteMeasurementEntropy = load_object("ChargeSharpeningThreeSiteL=10Entropies.jld2")


plot(problist, ETSingleSiteMeasurementEntropy[1], label="Single Site ET", linestyle=:solid)
plot!(problist, ETTwoSiteMeasurementEntropy[1], label="Two Site ET", linestyle=:dash)
plot!(problist, ETThreeSiteMeasurementEntropy[1], label="Three Site ET", linestyle=:dashdot)


scatter!(problist, ETSingleSiteMeasurementEntropy[1], label="")
scatter!(problist, ETTwoSiteMeasurementEntropy[1], label="")
scatter!(problist, ETThreeSiteMeasurementEntropy[1], label="")
xlabel!("p")
ylabel!("Ancilla Entropy")
title!("ET Ancilla Entropy L = 6")
png("ET Ancilla Entropy L = 6, 1000 trials, wide")

plot(problist, ETSingleSiteMeasurementEntropy[2], label="Single Site ET", linestyle=:solid)
plot!(problist, ETTwoSiteMeasurementEntropy[2], label="Two Site ET", linestyle=:dash)
plot!(problist, ETThreeSiteMeasurementEntropy[2], label="Three Site ET", linestyle=:dashdot)

plot!(problist, CSSingleSiteMeasurementEntropy[2], label="One Site CS", linestyle=:solid)
plot!(problist, CSTwoSiteMeasurementEntropy[2], label="Two Site CS", linestyle=:dash)
plot!(problist, CSL8ThreeSiteMeasurementEntropy[1], label="Three Site CS", linestyle=:dashdot)

scatter!(problist, ETSingleSiteMeasurementEntropy[2], label="")
scatter!(problist, ETTwoSiteMeasurementEntropy[2], label="")
scatter!(problist, ETThreeSiteMeasurementEntropy[2], label="")
xlabel!("p")
ylabel!("Ancilla Entropy")
title!("ET and CS Ancilla Entropy L = 8")
png("ET and CS Ancilla Entropy L = 8, 1000 trials, wide")

plot(problist, ETSingleSiteMeasurementEntropy[3], label="Single Site ET", linestyle=:solid)
plot!(problist, ETTwoSiteMeasurementEntropy[3], label="Two Site ET", linestyle=:dash)
plot!(problist, ETThreeSiteMeasurementEntropy[3], label="Three Site ET", linestyle=:dashdot)

plot!(problist, CSSingleSiteMeasurementEntropy[3], label="One Site CS", linestyle=:solid)
plot!(problist, CSTwoSiteMeasurementEntropy[3], label="Two Site CS", linestyle=:dash)
plot!(problist, CSL10ThreeSiteMeasurementEntropy[1], label="Three Site CS", linestyle=:dashdot)


scatter!(problist, ETSingleSiteMeasurementEntropy[3], label="")
scatter!(problist, ETTwoSiteMeasurementEntropy[3], label="")
scatter!(problist, ETThreeSiteMeasurementEntropy[3], label="")
xlabel!("p")
ylabel!("Ancilla Entropy")
title!("ET and CS Ancilla Entropy L = 10")
png("ET and CS Ancilla Entropy L = 10, 1000 trials, wide")



problist = LinRange(0.0, 0.5, 10)
OneSiteBipartate = load_object("L68EntanglementBipartateSingleSiteWideVariance.jld2")
TwoSiteBipartate = load_object("L68EntanglementBipartateTwoSiteWideVariance.jld2")
ThreeSiteBipartate = load_object("L68EntanglementBipartateThreeSiteWideVariance.jld2")

plot(problist, OneSiteBipartate[1], label="Single Site", linestyle=:solid)
plot!(problist, TwoSiteBipartate[1], label="Two Site", linestyle=:dash)
plot!(problist, ThreeSiteBipartate[1], label="Three Site", linestyle=:dashdot)
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Bipartate Entanglement Entropy Comparison L=8")
png("Bipartate Engtanglement Entropy MultiSite Comparison L810 ntrails1000")

plot(problist, OneSiteBipartate[2], label="Single Site", linestyle=:solid)
plot!(problist, TwoSiteBipartate[2], label="Two Site", linestyle=:dash)
plot!(problist, ThreeSiteBipartate[2], label="Three Site", linestyle=:dashdot)
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Bipartate Entanglement Entropy Comparison L = 10")
png("Bipartate Engtanglement Entropy MultiSite Comparison L10 ntrails1000")


