from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import pandas as pd
from typing import List, Dict, Any
from analyzer_model import ContentTrendsAnalyzer

#FastAPI
app = FastAPI(
    title="Content Trends API",
    description="API متقدمة لتحليل اتجاهات المحتوى وتوليد التوصيات",
    version="1.0.0"
)



class ContentItem(BaseModel):
    """نموذج Pydantic لصف واحد من بيانات المحتوى"""
    content: str = Field(..., description="نص المحتوى (مطلوب).")
    likes: int = Field(0, description="عدد الإعجابات.")
    num_comments: int = Field(0, description="عدد التعليقات.")
    num_shares: int = Field(0, description="عدد المشاركات.")
    num_hashtags: int = Field(0, description="عدد الهاشتاجات.")
    content_length: int = Field(0, description="طول المحتوى النصي.")
    media_type: str = Field("Unknown", description="نوع الوسائط (Video, Image, Text).")
    timestamp: str = Field(None, description="وقت النشر بصيغة ISO 8601 (مثل: '2025-10-25 14:30:00').")


class AnalysisRequest(BaseModel):
    """نموذج طلب التحليل الذي يحتوي على قائمة بـ ContentItem"""
    data: List[ContentItem] = Field(..., description="قائمة بالمنشورات المراد تحليلها.")


#(Endpoint)


@app.post("/analyze_trends", tags=["Analysis"])
def analyze_content_trends(request: AnalysisRequest):
    """
    تلقي بيانات المحتوى وإرجاع تقرير تحليل الاتجاهات والتوصيات.
    """
    if not request.data:
        raise HTTPException(status_code=400, detail="يجب توفير بيانات محتوى للتحليل.")

    try:
        # 1. تحويل بيانات الإدخال (قائمة من Pydantic Models) إلى DataFrame
        data_dicts = [item.model_dump() for item in request.data]
        df = pd.DataFrame(data_dicts)

        # 2. تهيئة المحلل
        analyzer = ContentTrendsAnalyzer(df)

        # 3. تشغيل التحليل
        report = analyzer.get_trending_analysis_report()

        # 4. إرجاع النتيجة (FastAPI يحولها تلقائيًا إلى JSON)
        return {
            "status": "success",
            "message": "تم التحليل بنجاح.",
            "report": report
        }

    except Exception as e:
        # التعامل مع أي خطأ قد يحدث أثناء التحليل
        raise HTTPException(
            status_code=500,
            detail=f"حدث خطأ داخلي أثناء معالجة التحليل: {e}"
        )


# نقطة نهاية اختبار بسيطة
@app.get("/", tags=["Health Check"])
def read_root():
    return {"message": "Content Trends API is running."}