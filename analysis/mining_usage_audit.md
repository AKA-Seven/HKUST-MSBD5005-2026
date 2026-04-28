# Mining Result Usage Audit

This visualization pipeline uses the following mining outputs and fields:

## outputs/q2/bundle_reliability.csv
- bundle, score, label
- seen_pair_ratio, valid_hscode_ratio, outside_date_ratio, physical_field_ratio, unique_pair_ratio
- ml_link_probability, fish_hscode_ratio, link_count

## outputs/q4/company_clusters.csv
- anomaly_score, isolation_label
- total_links, partner_count, hscode_count, fish_hscode_ratio
- dbscan_cluster, hierarchical_cluster
- nearest_company, nearest_similarity

## outputs/anomaly_delta.csv
- predicted_link_count
- filled_gap_months, extended_months, burst_months
- new_partner_count, new_hscode_count
- suspicious_revival_score

## outputs/q2/reliable_links.json
- source, target, arrivaldate
- generated_by, hscode, weightkg, valueofgoods_omu

The figures are intentionally designed so that each question uses both:
1) rule-based and ML-based reliability signals
2) structural and temporal anomaly signals
