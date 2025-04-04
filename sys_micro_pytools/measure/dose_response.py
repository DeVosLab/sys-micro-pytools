import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calibration_curve(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Calculate the calibration curve for a given dataframe.
    """
    print(df.head())
    # Fit the calibration curve
    model = LinearRegression().fit(df[[x_col]], df[[y_col]])

    return model

def calculate_ic_value(df, x_col, y_col, ic_percentile=50, threshold_pct=10, control_value=None):
        """
        Calculate IC value using either a 4-parameter logistic regression model or a quadratic model,
        selecting the one with better fit.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing dose-response data
        x_col : str
            Column name for dose values
        y_col : str
            Column name for response values
        control_value : float, optional
            Control value (no drug) to use as reference
        ic_percentile : float, optional
            Percentile of inhibition to calculate (e.g., 50 for IC50, 30 for IC30)
        threshold_pct : float, optional
            Minimum percent change required to consider a valid dose-response
            
        Returns:
        --------
        float
            IC value or None if no valid dose-response
        dict
            Parameters of the fitted model and quality metrics
        """
        from scipy.optimize import curve_fit
        import warnings
        
        def logistic_4pl(x, bottom, top, ic50, hill ):
            """4-parameter logistic function"""
            return bottom + (top - bottom) / (1 + (x / ic50) ** hill)
        
        def quadratic(x, a, b, c ):
            """Quadratic function"""
            return a * x**2 + b * x + c
        
        # Sort by x column
        df_sorted = df.sort_values(by=x_col)
        x_data = df_sorted[x_col].values
        y_data = df_sorted[y_col].values
        
        # Check if there's enough variation in the response
        y_min, y_max = min(y_data), max(y_data)
        y_range = y_max - y_min
        
        if control_value is not None:
            # Calculate percent inhibition relative to control
            max_inhibition_pct = 100 * abs(min(y_data) - control_value) / control_value
            if max_inhibition_pct < threshold_pct:
                return None, {
                    'status': 'insufficient_inhibition',
                    'max_inhibition_pct': max_inhibition_pct,
                    'threshold_pct': threshold_pct
                }
        elif y_range / y_max < threshold_pct/100:
            return None, {
                'status': 'flat_response',
                'y_range_pct': 100 * y_range / y_max,
                'threshold_pct': threshold_pct
            }
        
        # Try both fits
        fits = {}
        
        # 1. Try logistic fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0_logistic = [min(y_data), max(y_data), np.median(x_data), 1.0]
                bounds_logistic = (
                    [0, 0.5*max(y_data), min(x_data), -10],
                    [2*max(y_data), 2*max(y_data), 10*max(x_data), 10]
                )
                params_logistic, pcov_logistic = curve_fit(
                    logistic_4pl, x_data, y_data, 
                    p0=p0_logistic, bounds=bounds_logistic, maxfev=10000
                )
                
                y_pred_logistic = logistic_4pl(x_data, *params_logistic)
                ss_total = np.sum((y_data - np.mean(y_data))**2)
                ss_residual = np.sum((y_data - y_pred_logistic)**2)
                r_squared_logistic = 1 - (ss_residual / ss_total)
                
                fits['logistic'] = {
                    'params': params_logistic,
                    'pcov': pcov_logistic,
                    'r_squared': r_squared_logistic,
                    'model': logistic_4pl
                }
        except Exception as e:
            print(f'Error fitting logistic model: {e}')
            fits['logistic'] = None
            
        # 2. Try quadratic fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0_quad = [1, 1, np.mean(y_data)]
                params_quad, pcov_quad = curve_fit(quadratic, x_data, y_data, p0=p0_quad)
                
                y_pred_quad = quadratic(x_data, *params_quad)
                ss_total = np.sum((y_data - np.mean(y_data))**2)
                ss_residual = np.sum((y_data - y_pred_quad)**2)
                r_squared_quad = 1 - (ss_residual / ss_total)
                
                fits['quadratic'] = {
                    'params': params_quad,
                    'pcov': pcov_quad,
                    'r_squared': r_squared_quad,
                    'model': quadratic
                }
        except Exception as e:
            print(f'Error fitting quadratic model: {e}')
            fits['quadratic'] = None
            
        # Select best fit
        best_fit = None
        best_r_squared = -np.inf
        best_model = None
        
        for model_name, fit in fits.items():
            if fit is not None and fit['r_squared'] > best_r_squared:
                best_fit = fit
                best_r_squared = fit['r_squared']
                best_model = model_name
                
        if best_fit is None:
            return None, {'status': 'all_fits_failed'}
            
        # Calculate IC value based on best fit
        if best_model == 'logistic':
            bottom, top, ic50, hill = best_fit['params']
            response_level = bottom + (top - bottom) * (100 - ic_percentile) / 100
            
            if hill != 0:
                ic_value = ic50 * ((top - bottom) / (response_level - bottom) - 1) ** (1 / hill)
            else:
                ic_value = None
                
        else:  # quadratic
            a, b, c = best_fit['params']
            target_y = np.mean(y_data) * (100 - ic_percentile) / 100
            # Solve quadratic equation: ax² + bx + (c - target_y) = 0
            discriminant = b**2 - 4*a*(c - target_y)
            
            if discriminant >= 0:
                x1 = (-b + np.sqrt(discriminant)) / (2*a)
                x2 = (-b - np.sqrt(discriminant)) / (2*a)
                # Choose the root that's within our data range
                if min(x_data) <= x1 <= max(x_data):
                    ic_value = x1
                elif min(x_data) <= x2 <= max(x_data):
                    ic_value = x2
                else:
                    ic_value = None
            else:
                ic_value = None
        
        # Check if IC value is within the tested x range
        if ic_value is None:
            status = 'calculation_failed'
        elif ic_value < 0.5*min(x_data) or ic_value > 2*max(x_data):
            status = 'extrapolated_ic_value'
        elif best_r_squared < 0.7:
            status = 'poor_fit'
        else:
            status = 'good_fit'
            
        return ic_value, {
            'model_type': best_model,
            'params': best_fit['params'],
            'r_squared': best_r_squared,
            'status': status,
            'std_errors': np.sqrt(np.diag(best_fit['pcov'])),
            'model': best_fit['model']
        }

    