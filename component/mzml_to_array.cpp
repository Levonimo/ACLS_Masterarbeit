#include <OpenMS/FORMAT/MzMLFile.h>
#include <OpenMS/KERNEL/MSExperiment.h>
#include <cmath>
#include <vector>
#include <iostream>

static const double START = 35.0;
static const double END   = 400.0;
static const double STEP  = 0.1;

extern "C" {

void mzml_to_array(const char* file_path, double** output, int* rows, int* cols)
{
    OpenMS::MSExperiment exp;
    OpenMS::MzMLFile().load(file_path, exp);

    // Create a lookup for all possible m/z values (e.g., 35 to 400 with 0.1 step)
    std::vector<double> mZ_totlist;
    for (double d = START; d <= END + 1e-9; d += STEP)
    {
        mZ_totlist.push_back(std::round(d * 10) / 10);
    }

    // Collect spectra
    std::vector<std::vector<double>> chromatogramData;

    for (auto &spectrum : exp)
    {
        if (spectrum.getMSLevel() == 1)
        {
            auto mzArray = spectrum.getMZArray();
            auto intArray = spectrum.getIntensityArray();
            if (!mzArray || !intArray) 
                continue;

            const size_t size = mzArray->size();
            std::vector<double> full_intensity(mZ_totlist.size(), 0.0);

            // Bin each peak
            for (size_t i = 0; i < size; i++)
            {
                double mz_rounded = std::round((*mzArray)[i] * 10) / 10;
                // Find bin index via binary search or a direct mapping approach
                auto it = std::lower_bound(mZ_totlist.begin(), mZ_totlist.end(), mz_rounded);
                if (it != mZ_totlist.end() && *it == mz_rounded)
                {
                    size_t index = std::distance(mZ_totlist.begin(), it);
                    if (index < full_intensity.size())
                        full_intensity[index] = (*intArray)[i];
                }
            }
            chromatogramData.push_back(full_intensity);
        }
    }

    // Convert to a raw C-style 2D array (row-major)
    *rows = static_cast<int>(chromatogramData.size());
    *cols = static_cast<int>(mZ_totlist.size());
    *output = new double[(*rows) * (*cols)];

    for (int r = 0; r < *rows; ++r)
    {
        for (int c = 0; c < *cols; ++c)
        {
            (*output)[r * (*cols) + c] = chromatogramData[r][c];
        }
    }
}

} // extern "C"