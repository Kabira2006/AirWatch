"""
AirWatch Stakeholder Segmentation & RFM Exposure Profiling Module
Implements:
1. Operational Stakeholder Segmentation:
   - Segment A: Corporate Campuses & Tech Parks (HVAC triggers, productivity loss, WFH policy)
   - Segment B: Municipal & Urban Authorities (Ward heatmaps, traffic attribution, water misting)
   - Segment C: Vulnerable Institutions (Schools & Hospitals, pediatric/geriatric alerts, ER surge)
2. Geographic RFM (Risk, Frequency, Severity) Profiling for Urban Wards/Stations.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class WardRFMSegmentation:
    """
    Ranks urban wards and monitoring stations using an RFM framework:
    - R (Risk Baseline): Mean PM2.5, HVI, and population density exposure
    - F (Frequency of Violation): Percentage of days exceeding AQI > 150 & 200
    - S (Severity / Magnitude): Peak AQI and cumulative excess exposure beyond safety limits
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.path.join("data", "processed_sensor_business_data.csv")

    def load_clean_data(self) -> pd.DataFrame:
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
        elif os.path.exists("station_day_master.csv"):
            from backend.data_pipeline import SensorDataPipeline
            pipeline = SensorDataPipeline("station_day_master.csv")
            df = pipeline.run_pipeline(sample_size=30000)
        else:
            raise FileNotFoundError("No sensor data available for RFM calculation.")
        return df

    def compute_rfm_profiles(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculates RFM metrics per station/city ward.
        """
        if df is None:
            df = self.load_clean_data()

        group_col = "City" if "City" in df.columns else "StationId"

        records = []
        for ward, group in df.groupby(group_col):
            # 1. Risk Score (Baseline exposure)
            avg_aqi = group["AQI"].mean()
            avg_pm25 = group["PM2.5"].mean()
            hvi = group["Health_Vulnerability_Index"].mean() if "Health_Vulnerability_Index" in group.columns else avg_aqi / 3.0
            density = group["PopDensity_per_km2"].iloc[0] if "PopDensity_per_km2" in group.columns else 8000.0
            lat = group["Lat"].iloc[0] if "Lat" in group.columns else 20.0
            lon = group["Lon"].iloc[0] if "Lon" in group.columns else 78.0

            # 2. Frequency (Days with Unhealthy AQI > 150 and Severe > 200)
            total_days = len(group)
            days_unhealthy = (group["AQI"] > 150).sum()
            days_severe = (group["AQI"] > 200).sum()
            freq_unhealthy_pct = (days_unhealthy / total_days) * 100.0 if total_days > 0 else 0.0
            freq_severe_pct = (days_severe / total_days) * 100.0 if total_days > 0 else 0.0

            # 3. Severity / Magnitude (Peak AQI and Cumulative Excess Exposure over 100)
            max_aqi = group["AQI"].max()
            cumulative_excess = np.maximum(0, group["AQI"] - 100).sum()
            mean_excess_intensity = cumulative_excess / total_days if total_days > 0 else 0.0

            # Financial impact summary
            daily_ppl = group["Projected_Productivity_Loss_USD"].mean() if "Projected_Productivity_Loss_USD" in group.columns else avg_aqi * 12.0
            respiratory_risk = group["Respiratory_Complaint_Risk_pct"].mean() if "Respiratory_Complaint_Risk_pct" in group.columns else min(85.0, avg_aqi * 0.25)

            records.append({
                "Ward_or_City": ward,
                "Lat": round(lat, 4),
                "Lon": round(lon, 4),
                "Total_Observed_Days": total_days,
                "Avg_AQI": round(avg_aqi, 1),
                "Max_AQI": round(max_aqi, 1),
                "Avg_PM25": round(avg_pm25, 1),
                "PopDensity_per_km2": int(density),
                "Risk_HVI_Score": round(hvi, 2),
                "Freq_Unhealthy_pct": round(freq_unhealthy_pct, 1),
                "Freq_Severe_pct": round(freq_severe_pct, 1),
                "Severity_Excess_Intensity": round(mean_excess_intensity, 1),
                "Avg_Daily_PPL_USD": round(daily_ppl, 2),
                "Avg_Respiratory_Risk_pct": round(respiratory_risk, 1)
            })

        rfm_df = pd.DataFrame(records)

        # Min-max normalization for composite score
        r_norm = (rfm_df["Risk_HVI_Score"] - rfm_df["Risk_HVI_Score"].min()) / (rfm_df["Risk_HVI_Score"].max() - rfm_df["Risk_HVI_Score"].min() + 1e-6)
        f_norm = (rfm_df["Freq_Severe_pct"] - rfm_df["Freq_Severe_pct"].min()) / (rfm_df["Freq_Severe_pct"].max() - rfm_df["Freq_Severe_pct"].min() + 1e-6)
        s_norm = (rfm_df["Severity_Excess_Intensity"] - rfm_df["Severity_Excess_Intensity"].min()) / (rfm_df["Severity_Excess_Intensity"].max() - rfm_df["Severity_Excess_Intensity"].min() + 1e-6)

        # Composite RFM Exposure Index (0 to 100)
        rfm_df["RFM_Composite_Score"] = ((0.35 * r_norm + 0.35 * f_norm + 0.30 * s_norm) * 100.0).round(1)

        # Rank and Tier Assignment
        rfm_df = rfm_df.sort_values("RFM_Composite_Score", ascending=False).reset_index(drop=True)
        rfm_df["Exposure_Rank"] = rfm_df.index + 1

        percentiles = rfm_df["RFM_Composite_Score"].rank(pct=True)
        rfm_df["Intervention_Tier"] = np.select(
            [
                percentiles >= 0.75,
                (percentiles >= 0.50) & (percentiles < 0.75),
                (percentiles >= 0.25) & (percentiles < 0.50),
                percentiles < 0.25
            ],
            [
                "Tier 1: Critical Intervention Zone",
                "Tier 2: High Vigilance Zone",
                "Tier 3: Moderate Caution Zone",
                "Tier 4: Stable Baseline Zone"
            ],
            default="Tier 3: Moderate Caution Zone"
        )

        output_path = os.path.join("data", "ward_rfm_segmentation.csv")
        rfm_df.to_csv(output_path, index=False)
        print(f"RFM Segmentation completed! Profiles saved to: {output_path}")
        return rfm_df


class StakeholderDecisionEngine:
    """
    Translates raw and predicted air quality data into operational decisions
    across 3 strategic stakeholder personas:
    - Segment A: Corporate Campuses & Tech Parks
    - Segment B: Municipal & Urban Planning Authorities
    - Segment C: Vulnerable Institutions (Schools & Hospitals)
    """

    @staticmethod
    def get_corporate_advisory(predicted_aqi: float, current_aqi: float) -> Dict[str, Any]:
        """
        Segment A: Corporate Campuses & Tech Parks
        """
        effective_aqi = max(predicted_aqi, current_aqi)
        if effective_aqi <= 100:
            hvac_mode = "Fresh Air Economizer (Max Outdoor Intake)"
            filter_duty = "Low (Standard MERV-8)"
            work_policy = "Standard In-Office Operations"
            ppl_risk = "Negligible (< 1% payroll loss)"
            action_code = "CORP_NORMAL"
        elif effective_aqi <= 200:
            hvac_mode = "Mixed Air (50% Recirculation + MERV-13 Activation)"
            filter_duty = "Medium"
            work_policy = "Flexible Attendance; Sensitive Employees Remote Allowed"
            ppl_risk = f"Moderate (~$1,850/day per 1,000 workers)"
            action_code = "CORP_ELEVATED"
        elif effective_aqi <= 300:
            hvac_mode = "100% Internal Recirculation + Secondary Activated Carbon Filters"
            filter_duty = "High (HEPA + Carbon scrubbers)"
            work_policy = "Mandatory Hybrid/WFH for Non-Essential Staff"
            ppl_risk = f"Substantial (~$5,400/day per 1,000 workers)"
            action_code = "CORP_SEMI_LOCKDOWN"
        else:
            hvac_mode = "Emergency Building Pressurization & Continuous Scrubbing"
            filter_duty = "Maximum Overdrive (99.97% HEPA)"
            work_policy = "Full Campus Remote Work; Seal Entrance Vestibules"
            ppl_risk = f"Severe (~$11,200/day per 1,000 workers)"
            action_code = "CORP_FULL_SHUTDOWN"

        return {
            "Stakeholder_Segment": "Segment A: Corporate Campuses & Tech Parks",
            "Effective_AQI": round(effective_aqi, 1),
            "HVAC_Control_Trigger": hvac_mode,
            "Filtration_Duty_Cycle": filter_duty,
            "Workplace_Policy": work_policy,
            "Productivity_Risk_Impact": ppl_risk,
            "Action_Code": action_code
        }

    @staticmethod
    def get_municipal_advisory(ward_rfm: Dict[str, Any], forecasted_aqi: float) -> Dict[str, Any]:
        """
        Segment B: Municipal & Urban Planning Authorities
        """
        tier = ward_rfm.get("Intervention_Tier", "Tier 2: High Vigilance Zone")
        no2_level = ward_rfm.get("Avg_PM25", 80.0) * 0.45  # Estimated vehicular proxy

        if forecasted_aqi > 250 or "Tier 1" in tier:
            traffic_directive = "Emergency Freight Rerouting: Ban Diesel Trucks during 06:00-11:00"
            dust_suppression = "Continuous Anti-Smog Water Misting Cannon Deployment (Corridors A & C)"
            public_advisory = "Issue Public Smog Warning (Level-3 Municipal Emergency)"
            priority = "Immediate (Within 2 Hours)"
        elif forecasted_aqi > 150 or "Tier 2" in tier:
            traffic_directive = "Congestion Pricing Peak Surcharge; Enforce Synchronized Traffic Lights"
            dust_suppression = "Mechanical Road Sweeping & Targeted Water Sprinkling on Arterial Roads"
            public_advisory = "Issue Clean Air Advisory for Sensitive Groups"
            priority = "High (Within 6 Hours)"
        else:
            traffic_directive = "Normal Traffic Flow Monitoring"
            dust_suppression = "Scheduled Routine Street Cleansing"
            public_advisory = "Normal Air Quality Advisory"
            priority = "Routine"

        return {
            "Stakeholder_Segment": "Segment B: Municipal & Urban Authorities",
            "Ward_Zone": ward_rfm.get("Ward_or_City", "Metro Zone"),
            "Intervention_Tier": tier,
            "Traffic_Corridor_Directive": traffic_directive,
            "Municipal_Dust_Suppression": dust_suppression,
            "Public_Advisory_Level": public_advisory,
            "Response_Urgency": priority
        }

    @staticmethod
    def get_vulnerable_institution_advisory(forecasted_aqi: float) -> Dict[str, Any]:
        """
        Segment C: Vulnerable Institutions (Schools & Hospitals)
        """
        if forecasted_aqi <= 100:
            school_directive = "Normal Outdoor Recess & Athletic Activities Permitted"
            hospital_readiness = "Normal Respiratory Triage Bed Capacity"
            mask_mandate = "No Mask Requirement"
            alert_level = "GREEN: Low Vulnerability Risk"
        elif forecasted_aqi <= 200:
            school_directive = "Restrict Strenuous Outdoor Athletics; Move High-Cardio Sports Indoors"
            hospital_readiness = "Pre-stock Pediatric Nebulizer Stations (+15% surge buffer)"
            mask_mandate = "N95 / KF94 Recommended for Children with Asthma"
            alert_level = "YELLOW: Sensitive Group Advisory"
        elif forecasted_aqi <= 300:
            school_directive = "Total Cancellation of Outdoor Play & Field Assemblies; Seal Classroom Windows"
            hospital_readiness = "Activate Respiratory Emergency Surge Protocol (+35% Bed Buffer)"
            mask_mandate = "Mandatory N95 for All Students & Healthcare Visitors Outdoors"
            alert_level = "ORANGE: Acute Institutional Warning"
        else:
            school_directive = "Mandatory School Closure / Transition to Remote Learning Modules"
            hospital_readiness = "Code Red Pulmonary Surge: Deploy Supplemental Oxygen Reservoirs (+60%)"
            mask_mandate = "Strict Outdoor Exposure Prohibition for Pediatric & Geriatric Patients"
            alert_level = "RED: Severe Public Health Alert"

        return {
            "Stakeholder_Segment": "Segment C: Vulnerable Institutions (Schools & Hospitals)",
            "Forecasted_AQI": round(forecasted_aqi, 1),
            "School_Outdoor_Protocol": school_directive,
            "Hospital_ER_Surge_Readiness": hospital_readiness,
            "Personal_Protection_Mandate": mask_mandate,
            "Institutional_Alert_Level": alert_level
        }


if __name__ == "__main__":
    segmenter = WardRFMSegmentation()
    rfm_table = segmenter.compute_rfm_profiles()
    print("Top 5 Risk Wards:")
    print(rfm_table[["Ward_or_City", "Exposure_Rank", "RFM_Composite_Score", "Intervention_Tier"]].head())

    corp = StakeholderDecisionEngine.get_corporate_advisory(predicted_aqi=235.0, current_aqi=195.0)
    print("\nCorporate Advisory Sample:", corp)
