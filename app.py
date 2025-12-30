from flask import Flask, render_template, request
import joblib, os, pandas as pd, numpy as np
from werkzeug.utils import secure_filename
from flask import jsonify
# === CONFIGURATION FOR MODEL UPLOAD ===
UPLOAD_FOLDER = 'uploaded_models'
ALLOWED_EXTENSIONS = {'pkl'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    """Check if uploaded file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# === DEFAULT MODEL CONFIGURATION ===
model_files = {
    "Decision Tree": "decisiontree_all.pkl",
    "K-Nearest Neighbors (KNN)": "knn_all.pkl",
    "Naive Bayes": "naivebayes_all.pkl",
    "SVM Polynomial": "svm_mikroemulsi_best.pkl"
}

model_accuracy = {
    "SVM Polynomial": 100.0,
    "K-Nearest Neighbors (KNN)": 92.3,
    "Naive Bayes": 88.5,
    "Decision Tree": 88.5
}

default_model_name = "SVM Polynomial"

# === GLOBAL VARIABLES ===
current_model_path = None
current_model_name = default_model_name


@app.route("/", methods=["GET", "POST"])
def index():
    global current_model_path, current_model_name

    prediction = None
    form_data = {}
    error_msg = None

    selected_model_name = current_model_name
    model_path = current_model_path or model_files[default_model_name]

    # === CATEGORY DISPLAY MAPPING (MOVED UP) ===
    label_mapping = {
        "MINYAK": "Oil",
        "SURFAKTAN": "Surfactant",
        "LOGAM": "Metal",
        "KO_SURFAKTAN": "Co-Surfactant"
    }

    display_values = {
        "MINYAK": {
            "Minyak Sereh": "Lemongrass Oil",
            "Olive Oil": "Olive Oil",
            "Almond Oil": "Almond Oil",
            "Pine Oil": "Pine Oil",
            "Sweet Orange": "Sweet Orange Oil",
            "Castor Oil": "Castor Oil",
            "Virgin Coconut Oil": "Virgin Coconut Oil",
            "Minyak sawit": "Palm Oil",
            "Minyak Kelapa": "Coconut Oil"
        },
        "SURFAKTAN": {
            "Triton": "Triton",
            "Soya": "Soy Lecithin",
            "Heptanol": "Heptanol",
            "NP 10": "NP-10",
            "Tween 80": "Tween 80",
            "Coco Glucoside": "Coco Glucoside",
            "Span 80": "Span 80",
            "Peg 40": "PEG-40",
            "Decyl Glucoside": "Decyl Glucoside",
            "Peg 400": "PEG-400",
            "Ceteareth-25": "Ceteareth-25",
            "Laurid Acid": "Lauric Acid"
        },
        "LOGAM": {
            "Air": "Water",
            "Fe | Air": "Fe | Water",
            "Ni | Air": "Ni | Water",
            "Ekstrak Kulit Manggis": "Mangosteen Peel Extract",
            "Ekstrak Jeruk Lemon": "Lemon Extract",
            "Ekstrak Jahe": "Ginger Extract"
        },
        "KO_SURFAKTAN": {
            "Methanol": "Methanol",
            "Heptanol": "Heptanol",
            "Butanol": "Butanol",
            "Penthanol": "Pentanol"
        }
    }

    # === HANDLE MODEL UPLOAD OR MANUAL PATH INPUT ===
    if request.method == "POST":
        manual_path = request.form.get("model_path")

        if 'model_file' in request.files and request.files['model_file'].filename != '':
            file = request.files['model_file']
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                model_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(model_path)
                selected_model_name = filename
                current_model_path = model_path
                current_model_name = filename
            else:
                error_msg = "Model file must have a .pkl extension."
        elif manual_path and os.path.exists(manual_path):
            model_path = manual_path
            selected_model_name = os.path.basename(manual_path)
            current_model_path = model_path
            current_model_name = selected_model_name
        else:
            selected_model_name = request.form.get("model_choice", default_model_name)
            model_path = model_files.get(selected_model_name, model_files[default_model_name])
            current_model_name = selected_model_name
            current_model_path = model_path
    else:
        selected_model_name = current_model_name
        model_path = current_model_path or model_files[default_model_name]

    # === LOAD MODEL AND ENCODER ===
    try:
        model = joblib.load(model_path)
        le_y = joblib.load("label_encoder.pkl")
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        model = None
        le_y = None

    if not model:
        return render_template(
            "index.html",
            error_msg=error_msg,
            prediction=None,
            form_data=form_data,
            img_path=None,
            cat_values={},
            model_files=model_files,
            model_accuracy=model_accuracy,
            selected_model_name=selected_model_name,
            current_model_name=current_model_name
        )

    # === EXTRACT PREPROCESSOR INFO ===
    pre = None
    num_cols, cat_cols, expected_cols = [], [], []

    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        pre = model.named_steps["preprocessor"]
        for name, transformer, cols in pre.transformers_:
            if name == "num":
                num_cols = list(cols)
            elif name == "cat":
                cat_cols = list(cols)
        expected_cols = list(model.feature_names_in_)
    else:
        print("⚠️ Model does not contain preprocessor — direct prediction mode.")
        expected_cols = []

    # === PREDICTION PROCESS ===
    if request.method == "POST" and not error_msg:
        try:
            form_data = request.form.to_dict()

            # ==== HANDLE MANUAL INPUT VALUES ====
            def get_amount(field):
                # contoh field: 'JUMLAH_MINYAK'
                manual = request.form.get(f"{field}_MANUAL")
                slider = request.form.get(field)

                # Ambil manual kalau ADA (termasuk "0")
                if manual is not None and manual != "":
                    return manual

                # Kalau slider kosong, kembalikan "0"
                return slider if slider not in [None, ""] else "0"


            row = {}
            for c in expected_cols:
                if c in num_cols:
                    row[c] = form_data.get(c.replace(" ", "_"), form_data.get(c, 0))
                elif c in cat_cols:
                    row[c] = form_data.get(c.replace(" ", "_"), form_data.get(c, "Unknown"))
                else:
                    row[c] = form_data.get(c, "Unknown")

            # helper: aman untuk float
            def to_float(x):
                try:
                    if x is None:
                        return 0.0
                    x = str(x).strip().replace(",", ".")
                    return float(x) if x != "" else 0.0
                except:
                    return 0.0

            def get_logam_total():
                """
                Total JUMLAH LOGAM (g) versi baru:
                - kalau user mengisi JUMLAH_LOGAM (slider/manual) => pakai itu
                - kalau masih ada field legacy JUMLAH_LOGAM_1, JUMLAH_LOGAM_2, dst => jumlahkan (biar backward compatible)
                """
                # 1) legacy split inputs (kalau masih ada)
                split_vals = []
                i = 1
                while True:
                    v = request.form.get(f"JUMLAH_LOGAM_{i}")
                    if v is None:
                        break
                    split_vals.append(to_float(v))
                    i += 1

                if len(split_vals) > 0:
                    return float(sum(split_vals))

                # 2) normal single input
                return to_float(get_amount("JUMLAH_LOGAM"))

            logam_total = get_logam_total()

            row.update({
                "MINYAK": form_data.get("MINYAK"),
                "SURFAKTAN": form_data.get("SURFAKTAN"),
                "LOGAM": form_data.get("LOGAM"),
                "KO_SURFAKTAN": form_data.get("KO_SURFAKTAN"),

                # pastikan SEMUA numeric kolom model terisi
                "JUMLAH MINYAK (g)": get_amount("JUMLAH_MINYAK"),
                "JUMLAH SURFAKTAN (g)": get_amount("JUMLAH_SURFAKTAN"),
                "JUMLAH LOGAM (g)": logam_total,
                "JUMLAH KO-SURFAKTAN (g)": get_amount("JUMLAH_KO_SURFAKTAN"),
            })


            df_input = pd.DataFrame([row])

            for col in num_cols:
                if col in df_input.columns:
                    df_input[col] = pd.to_numeric(df_input[col], errors="coerce").astype("float64")

            if expected_cols:
                df_input = df_input.reindex(columns=expected_cols, fill_value=0)

            if df_input.isna().any().any():
                df_input = df_input.fillna(0)

            y_pred = model.predict(df_input)
            prediction = le_y.inverse_transform(y_pred)[0]

            # === Translate form_data for English history ===
            translated_form_data = {}
            translated_form_data["JUMLAH_LOGAM"] = str(logam_total)


            # translated_form_data["JUMLAH MINYAK (g)"] = row["JUMLAH MINYAK (g)"]
            # translated_form_data["JUMLAH SURFAKTAN (g)"] = row["JUMLAH SURFAKTAN (g)"]
            # translated_form_data["JUMLAH LOGAM (g)"] = row["JUMLAH LOGAM (g)"]
            # translated_form_data["JUMLAH KO-SURFAKTAN (g)"] = row["JUMLAH KO_SURFAKTAN (g)"]

            # ✅ Ambil nilai aktual (slider OR manual)
            translated_form_data["JUMLAH_MINYAK"] = get_amount("JUMLAH_MINYAK")
            translated_form_data["JUMLAH_SURFAKTAN"] = get_amount("JUMLAH_SURFAKTAN")
            translated_form_data["JUMLAH_KO_SURFAKTAN"] = get_amount("JUMLAH_KO_SURFAKTAN")

            # ✅ Khusus logam & logam split
            translated_form_data["JUMLAH_LOGAM"] = str(logam_total)


        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print("=== ERROR DURING PREDICTION ===", error_msg)

    # === RETRIEVE CATEGORY VALUES ===
    cat_values = {}
    if pre:
        try:
            for name, transformer, cols in pre.transformers_:
                if name == "cat":
                    ohe = transformer.named_steps.get("encoder")
                    if ohe:
                        for col, cats in zip(cols, ohe.categories_):
                            norm_col = col.upper().replace(" ", "_").replace("-", "_")
                            cat_values[norm_col] = list(cats)
        except Exception as e:
            print("=== WARNING: failed to extract categories ===", e)

    if not cat_values:
        cat_values = {
            "MINYAK": ["Minyak Sereh", "Olive Oil", "Almond Oil", "Pine Oil", "Sweet Orange", "Castor Oil",
                       "Virgin Coconut Oil", "Minyak sawit", "Minyak Kelapa"],
            "SURFAKTAN": ["Triton", "Soya", "Heptanol", "NP 10", "Tween 80", "Coco Glucoside", "Span 80",
                          "Peg 40", "Decyl Glucoside", "Peg 400", "Ceteareth-25", "Laurid Acid"],
            "LOGAM": ["Air", "Fe | Air", "Ni | Air", "Ekstrak Kulit Manggis", "Ekstrak Jeruk Lemon", "Ekstrak Jahe"],
            "KO_SURFAKTAN": ["Methanol", "Heptanol", "Butanol", "Penthanol"]
        }

    # === PREDICTION IMAGE ===
    img_path = None
    if prediction and prediction in ["1", "2", "3"]:
        fasa_images = {
            "1": "images/fasa/fasa1.jpg",
            "2": "images/fasa/fasa2.jpg",
            "3": "images/fasa/fasa3.jpg",
        }
        img_path = fasa_images[prediction]

    print("=== MODEL:", selected_model_name)
    print("=== Prediction:", prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        form_data=form_data,
        error_msg=error_msg,
        img_path=img_path,
        cat_values=cat_values,
        display_values=display_values,
        label_mapping=label_mapping,
        model_files=model_files,
        model_accuracy=model_accuracy,
        selected_model_name=selected_model_name,
        current_model_name=current_model_name,
        logam_display_mapping=display_values.get("LOGAM", {}),
    )

@app.route("/predict", methods=["POST"])
def predict_ajax():
    global current_model_path
    data = request.json

    try:
        model = joblib.load(current_model_path or model_files[default_model_name])
        le_y = joblib.load("label_encoder.pkl")
    except:
        return jsonify({"error": "Model gagal dimuat"}), 500

    row = pd.DataFrame([data])

    # =========================
    # 1️⃣ Pastikan kolom kategori (tidak memaksa ke string — hanya normalisasi nilai kosong/slider)
    # =========================
    cat_cols = ["MINYAK","SURFAKTAN","LOGAM","KO_SURFAKTAN","Ko-Surfaktan"]
    for col in cat_cols:
        if col in row.columns:
            val = row[col].iloc[0]
            # Jika datang sebagai string kosong / token slider default -> set ke None/np.nan
            if isinstance(val, str):
                v = val.strip()
                if v in ("", "0", "0.0", "None", "nan"):
                    row.at[0, col] = None
                else:
                    row.at[0, col] = v
            # jika sudah numeric, biarkan (jangan paksa menjadi string)

    # =========================
    # 2️⃣ Tambah alias kolom untuk model typo
    # =========================
    if "KO_SURFAKTAN" in row.columns:
        row["Ko-Surfaktan"] = row["KO_SURFAKTAN"]

    if "JUMLAH KO-SURFAKTAN (g)" in row.columns:
        row["JUMLAH KO-SURFAKTRAN (g)"] = row["JUMLAH KO-SURFAKTAN (g)"]

    # =========================
    # 3️⃣ Split logam multi → tetap string
    # =========================
    # if "LOGAM" in row.columns and "|" in str(row["LOGAM"].iloc[0]):
    #     vals = str(row["JUMLAH LOGAM (g)"].iloc[0]).split("|")
    #     vals = [v.strip() if v.strip() else "0" for v in vals]
    #     row["JUMLAH LOGAM (g)"] = " | ".join(vals)

    # =========================
    # 4️⃣ Paksa numeric jadi float
    # =========================
    num_cols = [
        "JUMLAH MINYAK (g)",
        "JUMLAH SURFAKTAN (g)",
        "JUMLAH LOGAM (g)",
        "JUMLAH KO-SURFAKTAN (g)"
    ]

    for col in num_cols:
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0).astype(float)

    print("INPUT FINAL:\n", row)

    # =========================
    # 5️⃣ Align input dtypes with encoder categories (avoid np.isnan on string arrays)
    # =========================
    try:
        preproc = model.named_steps.get("preprocessor")
        if preproc is not None:
            # cari transformer 'cat' dan nama kolomnya
            cat_cols_info = None
            cat_transformer = None
            for name, transformer, cols in preproc.transformers_:
                if name == "cat":
                    cat_cols_info = cols
                    cat_transformer = transformer
                    break

            # cari encoder di dalam pipeline transformer
            enc = None
            if cat_transformer is not None:
                if hasattr(cat_transformer, "named_steps"):
                    for key in ["encoder", "onehot", "ohe", "OneHotEncoder"]:
                        if key in cat_transformer.named_steps:
                            enc = cat_transformer.named_steps[key]
                            break
                else:
                    # transformer bisa jadi langsung OneHotEncoder
                    if type(cat_transformer).__name__.lower().startswith("onehotencoder"):
                        enc = cat_transformer

            if enc is not None:
                enc.handle_unknown = "ignore"

                # jika kita punya info kolom, sesuaikan tipe setiap kol input
                if cat_cols_info is not None:
                    for col_name, cats in zip(cat_cols_info, enc.categories_):
                        if col_name in row.columns:
                            try:
                                cats_arr = np.asarray(cats)
                                # deteksi apakah kategori numeric (semua elemen dapat diperlakukan sebagai angka)
                                is_numeric_cats = np.all([isinstance(c, (int, float, np.integer, np.floating)) or
                                                          (isinstance(c, str) and c.replace('.', '', 1).lstrip('-').isdigit())
                                                          for c in cats_arr])
                            except Exception:
                                is_numeric_cats = False

                            if is_numeric_cats:
                                # ubah token kosong/'None' -> NaN lalu konversi numeric
                                row[col_name] = row[col_name].replace({None: np.nan, "None": np.nan, "": np.nan})
                                row[col_name] = pd.to_numeric(row[col_name], errors="coerce")
                            else:
                                # pastikan string dan non-empty -> tetap string
                                row[col_name] = row[col_name].astype(object).where(row[col_name].notnull(), None)
    except Exception as e:
        print("🔥 OHE ALIGN FAILED:", e)

    # =========================
    # 6️⃣ Transform + Predict
    # =========================
    if "preprocessor" in model.named_steps:
        X = model.named_steps["preprocessor"].transform(row)
        pred = model.named_steps["classifier"].predict(X)
    else:
        pred = model.predict(row)

    phase = le_y.inverse_transform(pred)[0]

    img = {
        "1": "/static/images/fasa/fasa1.jpg",
        "2": "/static/images/fasa/fasa2.jpg",
        "3": "/static/images/fasa/fasa3.jpg"
    }.get(phase, "/static/images/fasa/default.png")


    return jsonify({"phase": phase, "image": img})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

