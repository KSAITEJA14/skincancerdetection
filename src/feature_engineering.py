import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def feature_engineering(df):

    data_copy = df.copy()
    
    try:
        data_copy.drop(columns=["patient_id", "lesion_id", "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5", "mel_mitotic_index", "mel_thick_mm", 
                     "tbp_lv_dnn_lesion_confidence","attribution", "copyright_license"], inplace=True)
    except KeyError:
        pass
    
    df = pd.DataFrame(data_copy)
    
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"] = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4
    
    df.replace([np.inf, -np.inf], -5.0, inplace=True)    
                   
    numerical_columns = ["target", "age_approx", "clin_size_long_diam_mm", "tbp_lv_A", "tbp_lv_Aext", "tbp_lv_B", "tbp_lv_Bext", "tbp_lv_C", "tbp_lv_Cext",
                         "tbp_lv_H", "tbp_lv_Hext", "tbp_lv_L", "tbp_lv_Lext", "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean", "tbp_lv_deltaA",
                         "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB", "tbp_lv_deltaLBnorm", "tbp_lv_eccentricity", "tbp_lv_minorAxisMM", "tbp_lv_nevi_confidence",
                         "tbp_lv_norm_border", "tbp_lv_norm_color", "tbp_lv_perimeterMM", "tbp_lv_radial_color_std_max", "tbp_lv_stdL", "tbp_lv_stdLExt", "tbp_lv_symm_2axis",
                         "tbp_lv_symm_2axis_angle", "tbp_lv_x", "tbp_lv_y", "tbp_lv_z", "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
                         "luminance_contrast", "lesion_color_difference", "border_complexity", "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio", "lesion_visibility_score",
                         "symmetry_border_consistency", "color_consistency", "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index", "shape_complexity_index",
                         "color_contrast_index", "log_lesion_area", "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast", "color_shape_composite_index", "3d_lesion_orientation",
                         "overall_color_difference", "symmetry_perimeter_interaction", "comprehensive_lesion_index"]
    
    categorical_columns = ["sex", "anatom_site_general", "image_type", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]
    
    id_columns = ["isic_id"]
    
    encoder = OrdinalEncoder()
    
    for col in df.columns:
        if col not in numerical_columns and col not in categorical_columns and col not in id_columns:
            df.drop(columns=[col], inplace=True)
    
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[[col]])
         
    return df