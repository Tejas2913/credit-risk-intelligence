-- ==============================================================================
-- Talk-to-Data Sample Business Queries
-- Verified SQL queries executable against applicant_analytics and relational tables
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- PART 1: SINGLE-TABLE ANALYTICAL QUERIES
-- ------------------------------------------------------------------------------

-- Query 1: Overall Portfolio Summary and Baseline Default Rate
-- Question: "What is the total number of applicants and the overall portfolio default rate?"
SELECT 
    COUNT(*) AS total_applicants,
    SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS total_defaults,
    ROUND(AVG(target) * 100, 2) AS default_rate_pct,
    ROUND(AVG(amt_income_total), 2) AS avg_annual_income,
    ROUND(AVG(amt_credit), 2) AS avg_credit_amount
FROM applicant_analytics;


-- Query 2: Default Rate and Average Loan Size by Education Level
-- Question: "How does the default rate and average credit amount vary across education levels?"
SELECT 
    name_education_type,
    COUNT(*) AS applicant_count,
    ROUND(AVG(target) * 100, 2) AS default_rate_pct,
    ROUND(AVG(amt_credit), 2) AS avg_credit_amount,
    ROUND(AVG(amt_income_total), 2) AS avg_income
FROM applicant_analytics
GROUP BY name_education_type
ORDER BY default_rate_pct DESC;


-- Query 3: Default Risk Segmented by Applicant Income Brackets
-- Question: "What is the default rate across different income tiers?"
SELECT 
    CASE 
        WHEN amt_income_total < 100000 THEN '1. Low Income (< 100k)'
        WHEN amt_income_total BETWEEN 100000 AND 200000 THEN '2. Middle Income (100k - 200k)'
        WHEN amt_income_total BETWEEN 200001 AND 350000 THEN '3. Upper Middle (200k - 350k)'
        ELSE '4. High Income (> 350k)'
    END AS income_tier,
    COUNT(*) AS applicant_count,
    ROUND(AVG(target) * 100, 2) AS default_rate_pct,
    ROUND(AVG(credit_income_ratio), 2) AS avg_credit_to_income
FROM applicant_analytics
GROUP BY income_tier
ORDER BY income_tier ASC;


-- Query 4: Impact of Historical Application Refusals on Default Risk
-- Question: "Are applicants who experienced previous loan refusals more likely to default?"
SELECT 
    CASE 
        WHEN prev_app_count = 0 THEN 'No Prior Applications'
        WHEN prev_refusal_rate > 0.5 THEN 'High Prior Refusals (>50%)'
        WHEN prev_refusal_rate > 0.0 THEN 'Some Prior Refusals (1-50%)'
        ELSE 'Clean Prior Approval History'
    END AS refusal_history_group,
    COUNT(*) AS applicant_count,
    ROUND(AVG(target) * 100, 2) AS default_rate_pct,
    ROUND(AVG(amt_credit), 2) AS avg_credit
FROM applicant_analytics
GROUP BY refusal_history_group
ORDER BY default_rate_pct DESC;


-- ------------------------------------------------------------------------------
-- PART 2: MULTI-TABLE RELATIONAL JOIN QUERIES
-- ------------------------------------------------------------------------------

-- Query 5: Relational Join - Applicants + Bureau Summary
-- Question: "Compare default rates for applicants with bureau debt above their credit limit."
SELECT 
    CASE 
        WHEN b.bureau_total_debt > a.amt_credit THEN 'Bureau Debt > Loan Amount' 
        WHEN b.bureau_total_debt > 0 THEN 'Bureau Debt <= Loan Amount' 
        ELSE 'Zero Bureau Debt' 
    END AS debt_exposure_tier, 
    COUNT(*) AS applicant_count, 
    ROUND(AVG(a.target) * 100, 2) AS default_rate_pct, 
    ROUND(AVG(b.bureau_total_debt), 2) AS avg_bureau_debt, 
    ROUND(AVG(a.amt_credit), 2) AS avg_loan_amount 
FROM applicants a 
JOIN bureau_summary b ON a.sk_id_curr = b.sk_id_curr 
GROUP BY debt_exposure_tier 
ORDER BY default_rate_pct DESC;


-- Query 6: Relational Join - Applicants + Installment Summary
-- Question: "What is the average historical installment payment ratio for defaulters vs non-defaulters?"
SELECT 
    CASE WHEN a.target = 1 THEN 'Defaulters (Target = 1)' ELSE 'Non-Defaulters (Target = 0)' END AS applicant_group, 
    COUNT(*) AS applicant_count, 
    ROUND(AVG(i.total_payment_amount), 2) AS avg_total_paid, 
    ROUND(AVG(i.late_payment_ratio) * 100, 2) AS avg_late_payment_pct 
FROM applicants a 
JOIN installment_summary i ON a.sk_id_curr = i.sk_id_curr 
GROUP BY applicant_group 
ORDER BY a.target DESC;


-- Query 7: Relational 3-Table Join - Applicants + Previous Applications + Installments
-- Question: "How many applicants had previous loan refusals and late installment payments?"
SELECT 
    CASE 
        WHEN p.prev_refusal_rate > 0.3 AND i.late_payment_ratio > 0.1 THEN 'High Refusal & High Late Payments' 
        WHEN p.prev_refusal_rate > 0.0 OR i.late_payment_ratio > 0.0 THEN 'Moderate Risk History' 
        ELSE 'Clean Prior History' 
    END AS combined_risk_profile, 
    COUNT(*) AS applicant_count, 
    ROUND(AVG(a.target) * 100, 2) AS default_rate_pct, 
    ROUND(AVG(a.amt_income_total), 2) AS avg_income 
FROM applicants a 
JOIN previous_application_summary p ON a.sk_id_curr = p.sk_id_curr 
JOIN installment_summary i ON a.sk_id_curr = i.sk_id_curr 
GROUP BY combined_risk_profile 
ORDER BY default_rate_pct DESC;


-- Query 8: Relational 3-Table Join - Applicants + Bureau + Previous Applications
-- Question: "Compare default rates between applicants with bureau history and previous application history."
SELECT 
    CASE 
        WHEN b.bureau_credit_count > 0 AND p.prev_app_count > 0 THEN 'Both Bureau & Prior App History' 
        WHEN b.bureau_credit_count > 0 THEN 'Bureau History Only' 
        WHEN p.prev_app_count > 0 THEN 'Prior App History Only' 
        ELSE 'No Prior Credit History' 
    END AS credit_history_profile, 
    COUNT(*) AS applicant_count, 
    ROUND(AVG(a.target) * 100, 2) AS default_rate_pct, 
    ROUND(AVG(a.amt_credit), 2) AS avg_credit 
FROM applicants a 
JOIN bureau_summary b ON a.sk_id_curr = b.sk_id_curr 
JOIN previous_application_summary p ON a.sk_id_curr = p.sk_id_curr 
GROUP BY credit_history_profile 
ORDER BY default_rate_pct DESC;
