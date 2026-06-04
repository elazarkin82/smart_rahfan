# Smart Rahfan - RKNN Integration & Optimization (RK3566 NPU)

תיקייה זו מרכזת את כלל הכלים, התסריטים (scripts) והמתודולוגיות לעבודה עם ה-NPU (Neural Processing Unit) של Rockchip על גבי כרטיס ה-**Radxa Zero 3** (המבוסס על שבב RK3566). ה-NPU מספק ביצועי חישוב של עד 1.0 TOPS ומצריך המרה של מודלים לפורמט הקנייני `.rknn` באמצעות **RKNN-Toolkit2**.

---

## 🎯 מטרות הפרויקט והארכיטקטורה

הפיתוח מחולק ל-4 ערוצים מרכזיים, אשר מיוצגים על ידי מבנה התיקיות הבא:

```text
rknn/
├── README.md                  # קובץ זה (תיעוד ראשי)
├── converter/                 # 1. כלי המרה סטנדרטיים (ONNX -> RKNN)
│   ├── convert.py             # סקריפט המרה גנרי עם הגדרות מותאמות ל-RK3566
│   └── configs/               # קובצי קונפיגורציה של המודלים השונים
│       └── model_config.yaml
│
├── inference/                 # 2. הרצת Inference מקומי
│   ├── pc_simulator.py        # הרצה ובדיקת ביצועים על ה-PC (RKNN Simulator)
│   └── board_inference.py     # הרצה על הלוח (Radxa Zero 3) בעזרת RKNN-Toolkit-Lite2
│
├── quantization/              # 3. אופטימיזציית קוונטיזציה (Quantization Intervention)
│   ├── calibrate.py           # הכנת קובצי כיול (Calibration Dataset) מתוך סט הנתונים
│   ├── quant_analyzer.py      # ניתוח שגיאות קוונטיזציה והשוואת דיוק (Cosine Similarity)
│   └── custom_quant.py        # התערבות בשלבי ה-Quantization של RKNN (שינוי פרמטרים ממוקדים)
│
└── graph_patcher/             # 4. החלפת אופרטורים לא נתמכים (Unsupported Ops Patcher)
    ├── patch_onnx.py          # סורק ומחליף אופרטורים ב-ONNX לפני ההמרה ל-RKNN
    └── rule_book.json         # חוקי החלפה (לדוגמה: החלפת אקטיבציות או אופרטורים מורכבים)
```

---

## 📁 1. כלי המרה (Converter Tools)
ההמרה מבוצעת על מחשב פיתוח (x86_64) בעזרת `rknn-toolkit2`.
*   **קלט:** מודל בפורמט `ONNX` (מומלץ לייצא מ-Keras/PyTorch בגרסת ONNX תואמת).
*   **פלט:** קובץ `.rknn` המותאם לריצה על ה-NPU.
*   **תכונות מתוכננות:**
    *   תמיכה בפרמטרים מוגדרים מראש עבור RK3566.
    *   אפשרות להמרה עם או בלי קוונטיזציה (INT8 לעומת FP16).
    *   תמיכה ב-Multi-input / Multi-output models.

---

## 💻 2. הרצת Inference מקומי (Inference Tools)
מניעת צווארי בקבוק בשלב הפיתוח על ידי הרצה בשני מצבים:
1.  **מצב סימולטור (PC Simulator):** בדיקת נכונות המודל וביצועי הדיוק שלו ישירות על מחשב הפיתוח (x86_64) ללא צורך בחיבור ללוח הפיזי.
2.  **הרצה על הלוח (Board Inference):** ריצה פיזית על ה-Radxa Zero 3 באמצעות `rknn-toolkit-lite2` (בפייתון) או ה-C API של Rockchip.
    *   מדידת זמני ריצה (Latency) וניצול זיכרון של ה-NPU בלבד.
    *   אינטגרציה עם מצלמת ה-MIPI או הזרמת UDP.

---

## ⚖️ 3. התערבות ואופטימיזציית קוונטיזציה (Quantization Optimization)
המרת מודל ל-INT8 עלולה לפגוע בדיוק שלו (במיוחד במודלי Tracking קטנים ומדויקים). נפתח כלים ייעודיים להתערבות בתהליך הקוונטיזציה:
*   **Calibration Dataset Pipeline:** יצירת סט נתונים מייצג לצורך כיול מדויק של טווחי האקטיבציה (Activation Ranges) של הרשת.
*   **Hybrid / Mixed Precision:** הגדרת שכבות רגישות (כגון שכבות ה-Output או שכבות מוקדמות) לעבוד ב-FP16 בזמן ששאר הרשת עובדת ב-INT8 (נתמך ע"י מנוע הקוונטיזציה של RKNN).
*   **Quantization Parameter Tuning:** הגדרת אופטימיזציות מתקדמות ב-RKNN API:
    *   `quantized_dtype`
    *   `quantized_algorithm` (כמו `normal` או `mmse`)
    *   שימוש ב-`optimization_level` מותאם.

---

## 🛠️ 4. טיפול באופרטורים לא נתמכים (Unsupported Operators Patcher)
ה-NPU של Rockchip תומך בסט ספציפי של אופרטורים (שכבות חישוב). שכבות מורכבות או מותאמות אישית (לדוגמה: אקטיבציות מתקדמות, Dynamic Slicing, או Reshape מורכב) עלולות להידחות או לעבור ריצה איטית ב-CPU.
*   **Static Graph Analysis:** סריקת קובץ ה-ONNX לפני ההמרה ואיתור נקודות תורפה שאינן נתמכות ב-RK3566 NPU.
*   **Automatic Network Patching:** החלפת תת-גרפים (Sub-graphs) שאינם נתמכים באלטרנטיבות יעילות ונתמכות NPU (לדוגמה: החלפת פונקציות אקטיבציה מסוימות ב-ReLU/Clip, או פירוק שכבות מורכבות לשכבות פשוטות יותר).
*   **Fallback Strategy:** במקרה שלא ניתן להחליף, המערכת תדע לבצע פיצול גרף אוטומטי (Graph Splitting) כך שהחלקים הנתמכים ירוצו על ה-NPU והחלקים שאינם נתמכים ירוצו ב-CPU של הלוח, תוך שמירה על רציפות ה-Inference.

---

## ⚙️ דרישות מערכת והתקנה ראשונית

### 1. סביבת פיתוח (x86_64 Host PC)
מומלץ להשתמש ב-Ubuntu 20.04/22.04 או סביבת Docker מתאימה.
```bash
# התקנת התלויות הבסיסיות
pip install onnx onnxruntime pyyaml

# התקנת RKNN-Toolkit2 (יש להוריד את ה-Wheel המתאים מהרפו הרשמי של airockchip)
# דוגמה:
pip install rknn-toolkit2-X.X.X-cp310-cp310-linux_x86_64.whl
```

### 2. סביבת הרצה על הלוח (Radxa Zero 3 - arm64)
*   יש לוודא שה-NPU מאופשר ב-Device Tree Overlay. ניתן לעשות זאת דרך `sudo rsetup` -> `Overlays` -> הפעלת `NPU` וביצוע Boot מחדש.
*   התקנת **RKNN-Toolkit-Lite2**:
    ```bash
    pip install rknn-toolkit-lite2-X.X.X-cp310-cp310-linux_aarch64.whl
    ```

---

## 🚀 שלבי עבודה מתוכננים (Next Steps)
1. **שלב א':** כתיבת ה-Converter הבסיסי (`converter/convert.py`) והרצת בדיקה ראשונית על מודל ONNX קיים.
2. **שלב ב':** יצירת תשתית בדיקת Inference בסימולטור המקומי (`inference/pc_simulator.py`).
3. **שלב ג':** פיתוח ה-Graph Patcher לצורך ניקוי והתאמת אופרטורים ב-ONNX.
4. **שלב ד':** הגדרת ממשק כיול ואופטימיזציית קוונטיזציה להתערבות מדויקת.
