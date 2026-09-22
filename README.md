# D0036E Lab 2: Linear and polynomial regression

A practical study of how age relates to mean income in 2020. The lab first fits a straight line to a limited age range, repeats the analysis across a wider age range, and then compares polynomial models.

The [notebook](Habil/lab2_regression.ipynb) contains the five lab tasks, results and three required figures. This README is a companion for learning the reasoning behind the code.

## Repository contents

| File | Purpose |
|---|---|
| `Habil/lab2_regression.ipynb` | Executed lab notebook with explanations, tables and plots |
| `inc_subset.csv` | 30 observations, ages 20–49, with mean income for 2020 |
| `inc_utf.csv` | 1,785 observations: 21 regions × 85 age labels |
| `requirements.txt` | Analysis packages and notebook interface |
| `.gitignore` | Excludes environments, caches, local settings and duplicate materials |

The course handout identifies SCB as the source of the supplied data. These CSV files are course-provided inputs; the repository does not retrieve live SCB data. The assignment PDF is not included. Income is expressed in dataset units because the supplied CSVs do not identify the currency scale.

## Set up and run

Python 3.12 was used for verification. From a terminal:

```sh
git clone git@github.com:LTU-HABIL/D0036E-LAB2.git
cd D0036E-LAB2
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```sh
source .venv/bin/activate
```

Install packages, register the kernel, and start JupyterLab:

```sh
python -m pip install -r requirements.txt
python -m ipykernel install --user --name d0036e-lab2 --display-name "Python (D0036E Lab 2)"
python -m jupyterlab
```

Open `Habil/lab2_regression.ipynb`, select **Python (D0036E Lab 2)**, and restart the kernel and run all cells. The notebook searches its working directory and parent directories for both CSV files, so it works from the repository root or the notebook folder. Keep the datasets inside the cloned repository.

If PowerShell blocks environment activation, use the environment's Python directly instead of changing your execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name d0036e-lab2 --display-name "Python (D0036E Lab 2)"
.\.venv\Scripts\python.exe -m jupyterlab
```

GitHub can display saved notebook outputs, but viewing them does not execute the code. Restarting and running all cells checks that the notebook works without variables left over from an earlier interactive session.

## Task 1: Loading and understanding the data

### Why write a loader?

The task requires basic file I/O rather than `pandas.read_csv` or Python's `csv` module. `load_csv_basic` therefore:

1. Opens the file with `utf-8-sig`, which also handles an optional UTF-8 byte-order mark.
2. Reads its text, splits it into lines, and skips blank lines.
3. Splits the header and each record at commas.
4. Uses `dict(zip(header, values))` to associate each field with its column name.
5. Converts selected fields with the supplied `field_types` mapping.
6. Returns a list of dictionaries, which pandas then stores as a DataFrame.

For example, `{"age": int, "2020": float}` converts age to an integer and income to a decimal number. The empty first header is renamed `source_index`; it is a saved row index, not a useful model feature.

The loader checks for empty files, repeated column names and unexpected field counts. It is deliberately simple: splitting at commas does not support quoted commas or embedded newlines. It is suitable for these supplied files, not a general-purpose CSV parser.

### Inspecting with a debugger and printouts

The lab explicitly requires **both** methods. Printing the count and sample records gives a quick, repeatable check. A debugger is better when you need to stop inside the loop and understand why a particular value is wrong.

In a notebook editor with debugging support:

1. Set a breakpoint at `row = dict(zip(header, values))` inside `load_csv_basic`.
2. Debug the loading cell and inspect `header` and `values`.
3. Step through conversion and inspect `row`. For the subset, `age` should become an integer and `2020` a float.
4. Repeat when the full dataset is loaded in Task 3. Its ages initially contain text, such as `16 years`; inspect the later cleaning step too.
5. Compare the values with the printed records.

Variables such as `header` and `values` are local to the function. After the function returns, inspect the returned list instead. The saved outputs do not establish that interactive debugger inspection has been performed.

## Task 2: Manual linear regression

### Feature, target and observations

The feature, $x$, is age. The target, $y$, is mean income in 2020. Each subset row represents an age-level average, not an individual person. The supplied subset contains ages **20–49**, even though the handout describes the range as 20–50.

### Why split the data?

The notebook uses 24 ages for training and 6 for validation. Training estimates the model parameters. Validation measures how closely the fitted model predicts ages it did not train on. Measuring only training error would give an optimistic assessment.

`np.random.default_rng(42)` produces a reproducible random ordering. The first 20% becomes validation data; the rest becomes training data. Sorting each part afterward only makes the tables easier to read—it does not change membership.

A random age split suits estimating missing ages within the observed range. It is less suitable for testing predictions beyond that range. Holding out the oldest ages would ask an extrapolation question instead. With just six validation ages, one split can give a fragile estimate. These are all 2020 observations, so the split does not test future-year forecasting.

### How least squares works

The model is

$$\hat y = a + bx.$$

Here $b$ is the slope and $a$ is the intercept. Least squares chooses them to minimize the sum of squared training errors. The vectorized implementation uses

$$b = \frac{n\sum x_i y_i - (\sum x_i)(\sum y_i)}{n\sum x_i^2 - (\sum x_i)^2},\qquad a = \bar y-b\bar x.$$

NumPy performs arithmetic over whole arrays: `x * y` multiplies corresponding elements, and `np.sum` adds them. This avoids a Python loop for each observation. The formula requires variation in age; if all ages were identical, the slope denominator would be zero. The provided data has many distinct ages.

The subset model is approximately

$$\hat y = -29.862003 + 9.919898x.$$

An extra year of age corresponds to about 9.92 income units in this fitted line. This is an association in the dataset, not evidence that getting one year older causes that income increase. The intercept refers to age zero, outside the analysed range, so it has little practical meaning here.

### Predictions, plots and MSE

For validation predictions, substitute each held-out age into the fitted equation. The scatter plot distinguishes training points from validation points, and the line shows the model across the age range.

The manually implemented mean squared error is

$$\operatorname{MSE}=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat y_i)^2.$$

For example, actual values `[10, 20]` and predictions `[12, 17]` have errors `[-2, 3]`, squared errors `[4, 9]`, and MSE `6.5`. Squaring prevents positive and negative errors from cancelling and gives larger errors more weight. The units are squared income units; MSE is not a percentage.

The subset validation MSE is **373.404**. The straight line captures the rising trend, but it misses the flattening toward the upper end. A score is best interpreted alongside the plot and prediction table, not called “good” solely because its numerical value appears small.

## Task 3: Full-data regression

### Cleaning and aggregation

`extract_leading_int` extracts the digits from labels such as `16 years`. The full data contains ages 16–99 and an open-ended `100+ years` group. Coding that last group as 100 is a lower-bound approximation, not a measured mean age.

`groupby("age")["2020"].mean()` averages the 21 regional values for each age, yielding 85 observations. Every region receives equal weight. A population-weighted national mean would require population counts that are not supplied. The model therefore describes an unweighted mean of regional income means.

The 80/20 split now produces 68 training ages and 17 validation ages. The model uses the same manual fitting and MSE functions as Task 2. The held-out set is called validation here; Task 3.4 of the handout calls it a test dataset.

### Why the straight line performs poorly

The fitted full-data model is approximately

$$\hat y = 325.144061 - 0.756280x.$$

Its negative slope does not mean income decreases at every age in the observations. It is the single least-squares compromise across a distribution that rises toward middle age and then falls. It overpredicts young ages and underpredicts much of the middle-age peak.

Validation MSE is **17,814.367**, much larger than the subset's **373.404**. The model is too inflexible for the full curve: this is **underfitting**. The two scores concern different validation ages and target distributions, so they do not show that adding data itself worsens a model.

## Task 4: Reflecting on the results

The restricted age range is approximately increasing, so a positive-slope line is a useful first approximation. The wider range has a pronounced peak that no single straight line can capture. Compare the graph, actual/predicted tables and MSE together to support that explanation.

A polynomial introduces curvature. Validation data can help choose how much flexibility to allow. Repeated splits or cross-validation could assess sensitivity to the held-out ages; they are possible improvements, not additional experiments implemented in this notebook.

## Task 5: Polynomial regression

### Why it is still linear regression

A degree-$d$ polynomial is

$$\hat y=b_0+b_1z+b_2z^2+\cdots+b_dz^d.$$

It curves as a function of age but remains linear in the fitted coefficients. `PolynomialFeatures` creates the powers; `LinearRegression` estimates their coefficients. The degree is a **hyperparameter** selected before fitting each candidate. The coefficients are **parameters** learned during fitting.

### Why standardize before creating powers?

Raw powers can differ enormously in size: $100^8=10^{16}$. This can make a high-degree least-squares fit numerically unstable. The pipeline first computes standardized age,

$$z=\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}},$$

then creates its powers. Scaling improves the numerical representation; it does not introduce regularization or change the family of degree-$d$ polynomial functions.

The pipeline contains:

1. `StandardScaler`: learns the mean and scale from training ages only.
2. `PolynomialFeatures`: creates powers through the chosen degree. `include_bias=False` omits a constant column because the regression already fits an intercept.
3. `LinearRegression`: fits coefficients using the transformed training data.

Calling `predict` applies the already learned transformations to validation ages. Fitting the scaler on validation data would leak information from evaluation into training. Using the same split for all candidates makes their MSE comparison consistent.

`full_train[["age"]]` uses double brackets to keep a two-dimensional feature table, as required by scikit-learn. The target is the one-dimensional `full_train["2020"]` series.

### Results and choice

| Model | Validation MSE |
|---|---:|
| Subset manual linear model | 373.404 |
| Full-data manual linear model | 17,814.367 |
| Full-data polynomial, degree 2 | 3,995.569 |
| Full-data polynomial, degree 3 | 259.964 |
| Full-data polynomial, degree 5 | 257.543 |
| Full-data polynomial, degree 8 | **186.570** |

Only the full-data models share the same validation observations. **Degree 8** wins among the four polynomial candidates for seed 42. `idxmin()` identifies the row with the smallest validation MSE, and the final plot overlays that model with the manual straight line.

Do not interpret the winning score as proof that degree 8 always generalizes best. The validation set was used to choose it, so its score is a tuning result rather than an independent final test estimate. Greater flexibility can also produce unrealistic predictions: at age 16, the selected model predicts approximately **−37.7**, whereas the observed mean is approximately **5.5**. A low overall MSE does not guarantee sensible predictions everywhere, and high-degree polynomials can behave poorly outside the observed range.

## What each library contributes

| Library | Role |
|---|---|
| `pathlib` | Locate the input files using filesystem paths |
| NumPy | Arrays, seeded shuffling, vectorized regression and MSE |
| pandas | Store parsed records, group by age and display tables |
| matplotlib | Draw the three regression figures |
| IPython | Display DataFrames in notebook output |
| scikit-learn | Scale age and fit polynomial models in Task 5 only |

Tasks 2 and 3 do not use scikit-learn for fitting or MSE. All model evaluations use the notebook's manual `mse` function.

## Self-check questions

- **Why avoid training MSE as the only evaluation?** The model was fitted to those same targets; held-out ages provide a more informative prediction check.
- **Why is age 100 special?** It represents the open-ended 100+ group rather than an exact single age.
- **Why does the full-data line slope downward?** It is a global compromise across a curved pattern, not a description of every local trend.
- **Why not always choose the highest degree?** Complexity may fit noise or behave badly at boundaries. Selection must use validation performance and interpretation.
- **Why scale inside a pipeline?** Training learns the transformation, and all subsequent inputs receive exactly that transformation.
- **What must be demonstrated interactively?** The debugger inspection of both CSV loads; saved printouts do not replace it.

## Reproducibility and troubleshooting

The notebook contains 12 executable cells and three figures. It was run from a fresh kernel in its own folder. Core analysis dependencies are pinned; JupyterLab is a compatible interface dependency rather than part of the numerical model. Small numerical differences can still occur across platforms.

- **Missing module:** install requirements in the environment used by the selected notebook kernel.
- **Data file not found:** launch within the repository and keep both CSVs at its root.
- **Undefined variable:** restart the kernel and run all cells in order.
- **Different results after editing:** restore seed 42 and the original candidate degrees, then rerun all cells. Markdown result summaries describe that configuration and need updating if the experiment changes.

Notebook outputs are committed so the results can be viewed on GitHub. Temporary checkpoints, virtual environments, local settings and duplicate downloads are ignored. After making a deliberate notebook change, rerun it before committing so saved results match the code.
