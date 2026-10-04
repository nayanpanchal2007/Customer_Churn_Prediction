from src.eda import run_eda
from src.train_models import train_all
from src.evaluate_models import save_model_comparison_charts, save_confusion_matrix, save_roc_curve
from src.explainability import permutation_feature_importance


def main():
    print("=== Customer Churn Prediction and Analysis System ===")
    print("1/4 Running EDA...")
    print(run_eda())

    print("2/4 Training classification models...")
    comparison, best_name, best_model, (X_test, y_test) = train_all()

    print("3/4 Generating evaluation artifacts...")
    save_model_comparison_charts(comparison)
    save_confusion_matrix(best_model, X_test, y_test, name="best_model")
    save_roc_curve(best_model, X_test, y_test, name="best_model")

    print("4/4 Generating explainability artifacts...")
    importance = permutation_feature_importance(best_model, X_test, y_test)

    print("\nModel comparison:")
    print(comparison.round(4))
    print(f"\nSelected model: {best_name}")
    print("\nTop permutation features:")
    print(importance.head(10).to_string(index=False))
    print("\nTraining completed. Start the dashboard with:")
    print("streamlit run app/app.py")


if __name__ == "__main__":
    main()
