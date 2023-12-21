import numpy as np
import toml
import sys
import pandas as pd

SECDAY = 86400e0


class WaterUse:
    def __init__(self):
        self.airtemp = 15.0
        # ambient air temperature (degrees Celsius)

        self.watertemp = 5.0
        # water temperature (degrees Celsius)

        self.windspeed = 4.0
        # wind speed (m/s)

        self.waterdens = 998.02
        # density of water (kg/m3)
        # for simplicity, set to static value applicable to water at 21 degC

        self.latentheat = 2257000.0
        # latent heat of vaporization (J/kg)
        # for simplicity, set to static value applicable at 1 atm (100 kPa) air pressure

        self.volheat = 0.0
        # volumetric heat of vaporization (J/m3)

        self.evapratio = 0.0
        # evaporation ratio: ratio of the heat used to evaporate water
        # to the total heat discharged through the tower

        self.evapvol = 0.0
        # evaporated volume of water (m3)

        self.energypervol = 0.0
        # input waste (heat) energy cooled per evaporated volume (J/m3)

        self.volperenergy = 0.0
        # volume evaporated by units of heat energy (m3/MJ)

        self.waterusetower = 0.0
        # total volume of water used in cooling tower (m3)

        self.wateruserecirc = 0.0
        # total volume of water used in recirculating system (m3)

        self.wateruseonethru = 0.0
        # total volume of water used in once-through system (m3)

    def run(self, pthermmw, etath):
        """
        Routine to call the water usage calculation routines.
        author: R Chapman, UKAEA

        This routine calls the different water usage routines.
        AEA FUS 251: A User's Guide to the PROCESS Systems Code

        Values taken from baseline 2017 output
        :param pthermmw: high-grade heat useful for electric production (MW)
        :type pthermw: float
        :param etath: thermal to electric conversion efficiency if `secondary_cycle=2`; otherwise calculated
        :type etath: float
        """
        rejected_heat = pthermmw * (1 - etath)

        wastethermeng = rejected_heat * SECDAY

        # Water usage during plant operation (secondary cooling)
        # Estimated amount of water used through different cooling system options:
        # 1. Cooling towers
        # 2. Water bodies (pond, lake, river): recirculating or once-through

        # call subroutines for cooling mechanisms:

        # cooling towers
        self.cooling_towers(wastethermeng)

        # water-body cooling
        self.cooling_water_body(wastethermeng)

    def cooling_towers(self, wastetherm: float):
        """Water used in cooling towers
        author: R Chapman, UKAEA
        wastetherm : input real : thermal energy (MJ) to be cooled by this system
        """
        self.evapratio = 1.0e0 - (
            (
                -0.000279e0 * self.airtemp**3
                + 0.00109e0 * self.airtemp**2
                - 0.345e0 * self.airtemp
                + 26.7e0
            )
            / 100.0e0
        )
        # Diehl et al. USGS Report 2013–5188, http://dx.doi.org/10.3133/sir20135188

        self.volheat = self.waterdens * self.latentheat

        self.energypervol = self.volheat / self.evapratio

        self.volperenergy = 1.0e0 / self.energypervol * 1000000.0e0

        self.evapvol = wastetherm * self.volperenergy

        # find water withdrawn from external source
        self.waterusetower = 1.4e0 * self.evapvol
        # Estimated as a ratio to evaporated water (averaged across obervered dataset)
        #  as per Diehl et al. USGS Report 2014–5184, http://dx.doi.org/10.3133/sir20145184

        # end break

        #  Output section
        # print(
        #     "Volume used in cooling tower (m3/day)",
        #     f"{self.waterusetower = :.3}",
        # )

    def cooling_water_body(self, wastetherm: float):
        """Water evaporated in cooling through water bodies
        Based on spreadsheet from Diehl et al. USGS Report 2013–5188, which includes
        cooling coefficients found through fits across a dataset containing a wide range of
        temperatures, windspeeds, and heat loading:
        http://pubs.usgs.gov/sir/2013/5188/appendix/sir2013-5188_appendix4_fews_version_3.104.xlsx

        author: R Chapman, UKAEA
        outfile : input integer : Fortran output unit identifier
        icool: input integer : switch between different water-body cooling options
        wastetherm : input real : thermal energy (MJ) to be cooled by this system
        """
        evapsum = 0.0e0

        for icool in range(1, 4):

            if icool == 1:
                # small pond as a cooling body
                # heat loading, MW/acre, based on estimations from US power plants
                heatload = 0.35e0
                # coefficients as per Brady et al. 1969:
                # wind function coefficients
                a = 2.47e0
                b = 0e0
                c = 0.12e0
                # fitted coefficients of heat loading
                d = 3061.331e0
                e = -48.810e0
                f = -78.559e0
                g = -291.820e0
                h = 0.267e0
                i = -0.610e0
                j = 33.497e0

            elif icool == 2:
                # large lake or reservoir as a cooling body
                # heat loading, MW/acre, based on estimations from US power plants
                heatload = 0.10e0
                # coefficients as per Webster et al. 1995:
                # wind function coefficients
                a = 1.04e0
                b = 1.05e0
                c = 0.0e0
                # fitted coefficients of heat loading
                d = 3876.843e0
                e = -49.071e0
                f = -295.246e0
                g = -327.935e0
                h = 0.260e0
                i = 10.528e0
                j = 40.188e0

            elif icool == 3:
                # stream or river as a cooling body
                # heat loading, MW/acre, based on estimations from US power plants
                heatload = 0.20e0
                # coefficients as per Gulliver et al. 1986:
                # wind function coefficients
                a = 2.96e0
                b = 0.64e0
                c = 0.0e0
                # fitted coefficients of heat loading
                d = 2565.009e0
                e = -43.636e0
                f = -93.834e0
                g = -203.767e0
                h = 0.257e0
                i = 2.408e0
                j = 20.596e0

            # Unfortunately, the source spreadsheet was from the US, so the fits for
            #   water body heating due to heat loading and the cooling wind functions
            #   are in non-metric units, hence the conversions required here.
            # Limitations: maximum wind speed of ~5 m/s; initial self.watertemp < 25 degC

            # convert self.windspeed to mph
            self.windspeedmph = self.windspeed * 2.237e0

            # convert heat loading into cal/(cm2.sec)
            heatloadimp = heatload * 1000000.0e0 * 0.239e0 / 40469000.0e0

            # estimate how heat loading will raise temperature, for this water body
            heatratio = (
                d
                + (e * self.watertemp)
                + (f * self.windspeedmph)
                + (g * heatload)
                + (h * self.watertemp**2)
                + (i * self.windspeedmph**2)
                + (j * heatload**2)
            )

            # estimate resultant heated water temperature
            self.watertempheated = self.watertemp + (heatloadimp * heatratio)

            # find wind function, m/(day.kPa), applicable to this water body:
            windfunction = (
                a + (b * self.windspeed) + (c * self.windspeed**2)
            ) / 1000.0e0

            # difference in saturation vapour pressure (Clausius-Clapeyron approximation)
            satvapdelta = (
                0.611e0
                * np.exp(
                    (17.27e0 * self.watertempheated) / (237.3e0 + self.watertempheated)
                )
            ) - (
                0.611e0
                * np.exp((17.27e0 * self.watertemp) / (237.3e0 + self.watertemp))
            )

            # find 'forced evaporation' driven by heat inserted into system
            deltaE = self.waterdens * self.latentheat * windfunction * satvapdelta

            # convert heat loading to J/(m2.day)
            heatloadmet = heatload * 1000000.0e0 / 4046.85642e0 * SECDAY

            # find evaporation ratio: ratio of the heat used to evaporate water
            #   to the total heat discharged through the tower
            self.evapratio = deltaE / heatloadmet
            # Diehl et al. USGS Report 2013–5188, http://dx.doi.org/10.3133/sir20135188

            self.volheat = self.waterdens * self.latentheat

            self.energypervol = self.volheat / self.evapratio

            self.volperenergy = 1.0e0 / self.energypervol * 1000000.0e0

            self.evapvol = wastetherm * self.volperenergy

            # using this method the estimates for pond, lake and river evaporation produce similar results,
            #   the average will be taken and used in the next stage of calculation
            evapsum = evapsum + self.evapvol

        evapsum = evapsum / icool

        # water volume withdrawn from external source depends on recirculation or 'once-through' system choice
        #   Estimated as a ratio to evaporated water (averaged across obervered dataset)
        #   as per Diehl et al. USGS Report 2014–5184, http://dx.doi.org/10.3133/sir20145184

        # recirculating water system:
        self.wateruserecirc = 1.0e0 * evapsum

        # once-through water system:
        self.wateruseonethru = 98.0e0 * evapsum

        # end break

        #  Output section
        # print(
        #     "Volume used in recirculating water system (m3/day)",
        #     f"{self.wateruserecirc = :.3}",
        # )
        # print(
        #     "Volume used in once-through water system (m3/day)",
        #     f"{self.wateruseonethru = :.3}",
        # )


def main():
    # Read in toml file
    toml_filename = sys.argv[1]
    inputs = toml.load(toml_filename)
    # print(f"{inputs = }")

    # Set as module variables for now
    pthermmw = inputs["pthermmw"]
    etath = inputs["etath"]

    # Run model
    water_use = WaterUse()
    water_use.run(pthermmw, etath)

    # Output csv file
    df = pd.DataFrame({"water_use_tower": water_use.waterusetower}, index=[0])
    df.to_csv("out.csv", index=False)


if __name__ == "__main__":
    main()
