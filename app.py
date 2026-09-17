import streamlit as st
import pandas as pd
import joblib

# ---------- إعدادات الصفحة ----------
st.set_page_config(page_title="توقع عيوب التصنيع", page_icon="🏭", layout="centered")

# ---------- تحميل الموديل ----------
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names

model, feature_names = load_model()

st.title("🏭 توقع عيوب التصنيع")
st.write("أدخل بيانات الإنتاج والماتيريالز عشان تعرف احتمالية ظهور عيب، ونسبة التأكد من التوقع.")

st.divider()

# ---------- المدخلات ----------
# كل عمود مع (min, max, default) مبنية على نطاق الداتا الحقيقي
col1, col2 = st.columns(2)

with col1:
    production_volume = st.slider("Production Volume (عدد الوحدات)", 100, 1000, 550)
    production_cost = st.slider("Production Cost ($)", 5000, 20000, 12000)
    supplier_quality = st.slider("Supplier Quality (%)", 80.0, 100.0, 90.0)
    delivery_delay = st.slider("Delivery Delay (أيام)", 0, 5, 2)
    defect_rate = st.slider("Defect Rate (تاريخي)", 0.5, 5.0, 2.7)
    quality_score = st.slider("Quality Score (%)", 60.0, 100.0, 80.0)
    maintenance_hours = st.slider("Maintenance Hours", 0, 23, 11)
    downtime_pct = st.slider("Downtime Percentage (%)", 0.0, 5.0, 2.5)

with col2:
    inventory_turnover = st.slider("Inventory Turnover", 2.0, 10.0, 6.0)
    stockout_rate = st.slider("Stockout Rate", 0.0, 0.1, 0.05)
    worker_productivity = st.slider("Worker Productivity (%)", 80.0, 100.0, 90.0)
    safety_incidents = st.slider("Safety Incidents", 0, 9, 5)
    energy_consumption = st.slider("Energy Consumption (kWh)", 1000, 5000, 3000)
    energy_efficiency = st.slider("Energy Efficiency", 0.1, 0.5, 0.3)
    additive_process_time = st.slider("Additive Process Time (ساعات)", 1.0, 10.0, 5.5)
    additive_material_cost = st.slider("💰 Additive Material Cost ($)", 100, 500, 300)

st.divider()

# ---------- التوقع ----------
if st.button("🔍 احسب التوقع", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([[
        production_volume, production_cost, supplier_quality, delivery_delay,
        defect_rate, quality_score, maintenance_hours, downtime_pct,
        inventory_turnover, stockout_rate, worker_productivity, safety_incidents,
        energy_consumption, energy_efficiency, additive_process_time, additive_material_cost
    ]], columns=feature_names)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    confidence = max(proba) * 100

    if prediction == 1:
        st.error(f"⚠️ توقع: **فيه احتمال عيب في الإنتاج**")
    else:
        st.success(f"✅ توقع: **الإنتاج سليم غالبًا (مفيش عيب متوقع)**")

    st.metric("نسبة التأكد من التوقع", f"{confidence:.1f}%")

    st.progress(confidence / 100)

    # ---------- نصيحة بسيطة بناءً على أهم الفيتشرز ----------
    st.subheader("📋 ملاحظات")
    notes = []
    if maintenance_hours > 15:
        notes.append("- ساعات الصيانة مرتفعة نسبيًا — من أقوى العوامل المؤثرة في ظهور العيوب.")
    if defect_rate > 3.5:
        notes.append("- معدل العيوب التاريخي لهذه الماتيريالز مرتفع.")
    if quality_score < 75:
        notes.append("- Quality Score منخفض نسبيًا — عامل مهم في تقليل العيوب.")
    if not notes:
        notes.append("- القيم المدخلة في نطاق آمن نسبيًا بناءً على بيانات المصنع.")
    for n in notes:
        st.write(n)

st.divider()
st.caption("الموديل: Random Forest مدرّب على بيانات إنتاج تاريخية. النتائج تقديرية وليست بديلاً عن الفحص الفعلي.")
