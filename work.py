import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")


menu=["Home","More on Heart Disease","About","Prediction"]
submenu=["Predictive Analytics"]

feature_columns_best=["BMI","Smoking","AlcoholDrinking","Stroke","DiffWalking","Sex","AgeCategory","Diabetic",
                  "PhysicalActivity","KidneyDisease"]

gender_dict= {"Female":1,"Male":2}
features_dict={"No":1,"Yes":2}
age_category_dict={"18-24":1,"25-29":2,"30-34":3,"35-39":4,"40-44":5,"45-49":6,"50-54":7,"55-59":8,"60-64":9,
              "65-69":10,"70-74":11,"75-79":12,"80 or older":13}

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return key
        
def get_fvalue(val):
    features_dict={"No":1,"Yes":2}
    for key,value in features_dict.items():
        if val == key:
            return value
        
#loading the machine learning models
def load_model(model_file):
    loaded_model=joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    choice=st.sidebar.selectbox("Menu",menu)
    st.sidebar.write(choice)

    if choice=="Home": 
        # giving the webpage a title
	    # here we define some of the front end elements of the web page like
	    # the font and background color, the padding and the text to be displayed
        html_temp = """
	    <div style ="background-color:yellow">
	    <h2 style ="color:black;text-align:center;">Heart Disease Prediction App </h2>
	    </div>
	    """
 	    # this line allows us to display the front end aspects we have
	    # defined in the above code
        st.markdown(html_temp,unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")

        st.write("This is a Machine Learning web application that helps to identify if a patient is suffering from Heart Disease related problems.",
                 "This application will help doctors to diagonose patient quickly thereby reducing the number of deaths related to Heart diseases.")
        
        with st.container():
            st.write("Heart disease refers to any condition affecting the heart.\t",
                 "According to the Centers for Disease Control and Prevention (CDC), heart disease is the leading cause of death in the United States and the condition affects all genders.\n",
                 'Heart disease refers to any condition affecting the cardiovascular system.\t',
                 'There are several different types of heart disease, and they affect the heart and blood vessels in different ways.\t')  
             
        symptom_temp = """
        <p style ="color:green;text-align:center;">Symptoms of Heart Disease</p>
        """
        st.markdown(symptom_temp,unsafe_allow_html=True)
        
        with st.container():
            st.write("\n")
            st.write("The symptoms of heart disease depend on the specific type a person has. Also, some heart conditions cause no symptoms at all.\n",
                 "That said, the following symptoms may indicate a heart problem:\n",
                 "1. Angina, or chest pain\n",
                 "2. Difficulty breathing\n",
                 "3. Fatigue and Lightheadedness\n",
                 "4. Swelling due to fluid retention, or edema\n",
                 "5. In children, the symptoms of a congenital heart defect may include cyanosis, or a blue tinge to the skin, and an inability to exercise.\n",
                 "6. Heart palpitations\n",
                 "7. Nausea\n",
                 "8. Stomach pain and sweating\n",
                 "9. Arm, jaw, back, or leg pain")             
                 
    elif choice=="More on Heart Disease":
        st.write("In this section, we shall look at some of the risks factors that cause a patient to be very prone to these problems.",
             "A detailed explanation about these risk factors is provided below as it helps to educate the patients what they can avoid.")
        st.subheader("Heart Disease Problems Risk Factors")
        with st.container():
            st.write("The heart disease problems have some risk factores which include:\n",
                 "1. Age. Growing older increases your risk of damaged and narrowed arteries and a weakened or thickened heart muscle.\n",
                 "2. Sex. Men are generally at greater risk of heart disease. The risk for women increases after menopause.\n",
                 "3. Family history. A family history of heart disease increases your risk of coronary artery disease, especially if a parent developed it at an early age (before age 55 for a male relative, such as your brother or father, and 65 for a female relative, such as your mother or sister).\n",
                 "4. Smoking. Nicotine tightens your blood vessels, and carbon monoxide can damage their inner lining, making them more susceptible to atherosclerosis. Heart attacks are more common in smokers than in nonsmokers.\n",
                 "5. Poor diet. A diet that's high in fat, salt, sugar and cholesterol can contribute to the development of heart disease.\n",
                 "6. High blood pressure. Uncontrolled high blood pressure can result in hardening and thickening of your arteries, narrowing the vessels through which blood flows.\n",
                 "7. High blood cholesterol levels. High levels of cholesterol in your blood can increase the risk of plaque formation and atherosclerosis.\n",
                 "8. Diabetes. Diabetes increases your risk of heart disease. Both conditions share similar risk factors, such as obesity and high blood pressure.\n",
                 "9. Obesity. Excess weight typically worsens other heart disease risk factors.\n",
                 "10. Physical inactivity. Lack of exercise also is associated with many forms of heart disease and some of its other risk factors as well.\n",
                 "11. Stress. Unrelieved stress may damage your arteries and worsen other risk factors for heart disease.\n",
                 "12. Poor dental health. It's important to brush and floss your teeth and gums often, and have regular dental checkups. If your teeth and gums aren't healthy, germs can enter your bloodstream and travel to your heart, causing endocarditis.\n",
                 "The above are some of the risk factors that causes a patient to be prone to these disease.")
        
    elif choice=="About":  
        about_temp = """
        <h3 style ="color:blue;text-align:center;">About this application</h3>
         """
        st.markdown(about_temp,unsafe_allow_html=True)
        with st.container():
            st.write("\n")
            st.write("\n")
            st.write("This is a machine learning web application that helps identify if a patient has Heart Disease related problems or not.\n"
                     "This can be helpful in preventing heart problems such as Heart Attacks which commonly occur at a sudden without being diagnosed.", 
                     "This application uses the patients information to detect if a patient has heart or the patient is prone to heart disease.",
                     "Using this information the machine learning algorithms makes the prediction on the patients data and poduces the output.",
                     "The output is NO meaning that the patient has No heart disease related problems and YES for patients having heart disease related problems.")
            st.write("\n")
            st.write("This application can be in corporated in hospital systems as this will assist doctors in Diagnosing",
                     "heart disease problems to patients.")
            st.write("This will also help patients know their status about heart diseases and identify those who are easily affected by this attacks.",
                     "The system also gives recommendations to those patients with this problems and advices them on how they can live a healthy lifestyle.",
                     "In case of extreme conditions, the application gives recommendations to doctors on how they can help the patient.")
            
            st.write("\n")
            st.write("Kindly interact with it!!")
    else:
        activity=st.sidebar.selectbox("activity",submenu)
        if activity=="Predictive Analytics":
            pred_temp = """
            <div style ="background-color:green">
            <h3 style ="color:white;text-align:center;">Predictive Analytics</h3>
            </div>
            """
            st.markdown(pred_temp,unsafe_allow_html=True)
            st.write("\n")
            st.write("\n")
            Sex=["Female","Male"]
            Smoking=("Yes","No")
            AlcoholDrinking=["Yes","No"]
            Stroke=["Yes","No"]
            DiffWalking=["Yes","No"]
            AgeCategory=["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80 or older"]
            Diabetic=["Yes","No","No, borderline diabetes","Yes (during pregnancy)"]
            PhysicalActivity=["Yes","No"]
            KidneyDisease=["Yes","No"]

            Sex=st.radio("Sex", tuple(gender_dict.keys())) 
            with st.form(key="BMIform"):
                col1, col2, col3 =st.columns([3,2,1])
                
                with col1:
                    weight=st.number_input("Enter your weight in Kilograms",min_value=20,max_value=200) 
                    
                with col2:
                    height=st.number_input("Enter your height in Centimetres",min_value=60,max_value=200)
                        
                with col3:
                    st.text("BMI")
                    bmi_calculation=st.form_submit_button(label="Calculate")
            BMI= np.round((weight/height),2)
            if bmi_calculation:
                with st.expander("BMI"):
                    BMI= (weight/height)
                    st.write(np.round(BMI,2))
                    
                if (BMI <=18.5):
                    st.warning("You are an Underweight and there is need to control this situation!!!")
                elif (BMI> 18.6) & (BMI <=24.9):
                    st.success("You are in a Normal Weight Condition!!")
                elif (BMI >=25.0) &  (BMI <=29.9):
                    st.warning("You are an Overweight Person. Take some action about this condition!!!")
                else:
                    st.warning("You are an Obese person. This condition increases your chances of getting Heart Disease problems!!")

            Smoking=st.selectbox("Do you smoke?",tuple(features_dict.keys()))
            st.write("\n")
            AlcoholDrinking=st.selectbox("Do you normally Drink Alcohol?", tuple(features_dict.keys()))
            st.write("\n")
            stroke=st.selectbox("Have you ever suffered stroke?", tuple(features_dict.keys()))
            st.write("\n")
            diffWalking=st.selectbox("Do you Diffwalk", tuple(features_dict.keys()))
            st.write("\n")
            AgeCategory=st.selectbox("What is your Age category", tuple(age_category_dict.keys()))
            st.write("\n")
            Diabetic=st.selectbox("Are you Diabetic",tuple(features_dict.keys()))
            st.write("\n")
            PhysicalActivity =st.selectbox("Do you usually Exercise", tuple(features_dict.keys()))
            st.write("\n")
            KidneyDisease=st.selectbox("Do you have Kidney Problems", tuple(features_dict.keys()))
            st.write("\n")
            
            feature_list=[get_value(Sex,gender_dict),BMI,get_fvalue(Smoking),get_fvalue(AlcoholDrinking),get_fvalue(stroke),
                      get_fvalue(diffWalking), get_value(AgeCategory,age_category_dict),get_fvalue(Diabetic),
                      get_fvalue(PhysicalActivity),get_fvalue(KidneyDisease)]
        
            st.write(feature_list)
        
            results={"Sex":Sex,"BMI":BMI,"Smoking":Smoking,"AlcoholDrinking":AlcoholDrinking,"stroke":stroke,"diffwalking":diffWalking,
                 "AgeCategory":AgeCategory,"Diabetic":Diabetic,"PhysicalActivity":PhysicalActivity,"KidneyDisease":KidneyDisease}
            st.json(results)
            input_transformed=np.array(feature_list).reshape(1, -1)
        
            #user to choice the model to use for prediction
            choice_of_model=st.selectbox("Choose the Machine Learning model",["KnearestNeighbors","LogisticRegression",
                                        "Decision Tree","ExtraTrees","Adaboost","Gradient Boosting"])
            st.write("\n")
            if st.button('Predict your Condition'):
                if choice_of_model == "KnearestNeighbors":
                    loaded_model = load_model("Models\KnearestNeighbors.pkl")
                    predictions=loaded_model.predict(input_transformed)
                    prediction_probability=loaded_model.predict_proba(input_transformed)
                
                elif choice_of_model == "Decision Tree":
                    loaded_model = load_model("Models\DecisionTreeModel.pkl")
                    predictions=loaded_model.predict(input_transformed)
                    prediction_probability=loaded_model.predict_proba(input_transformed)
                
                
                elif choice_of_model =="ExtraTrees":
                    loaded_model = load_model("Models\ExtraTreesModel.pkl")
                    predictions=loaded_model.predict(input_transformed)
                    prediction_probability=loaded_model.predict_proba(input_transformed)
                    
                elif choice_of_model =="Adaboost":
                    loaded_model = load_model("Models\AdaboostClassifier.pkl")
                    predictions=loaded_model.predict(input_transformed)
                    prediction_probability=loaded_model.predict_proba(input_transformed)
                    
                else:
                    loaded_model = load_model("Models\GradientBoostingClassifier.pkl")
                    predictions=loaded_model.predict(input_transformed)
                    prediction_probability=loaded_model.predict_proba(input_transformed)
                    
                
                st.write(predictions)
                if predictions == "Yes":
                    st.warning("The patient has Heart Disease Related Problems")
                    prescriptive_temp = """ 
                    <div style ="background-color:black">
                    <h3 style ="color:red;text-align:center;">Prescriptive Analytics</h3>
                    </div>
                    """
                    st.markdown(prescriptive_temp,unsafe_allow_html=True)
                    st.write("\n")
                    st.write("\n")
                    with st.container():  
                        recom_temp = """ 
                        <p style ="color:green;text-align:center;">Recommended Life style for Health Living</p>
                        """
                        st.markdown(recom_temp,unsafe_allow_html=True)
                        st.write("1. Increase Physical Activity: Moving more can lower your risk factors for heart disease. It does this by reducing the chances of developing high blood pressure. \n")
                        st.write("2. Eat a healthy balanced Diet. Eating a healthy diet is the key to heart disease prevention.\n")
                        st.write("3. Mantain a healthy weight: Mantaining a healthy weight is important for your heart health.\n")
                        st.write("4. Quit Smoking: Smoking is a major risk factor for developing atherosclerosis.   Quitting smoking will reduce your risk 0f developing heart disease problems.\n")
                        st.write("5. Reduce Alcohol Consumption. Always avoid binge drinking, as this increases the risk of a heart attack.\n")
                        st.write("6. Keep your blood pressure under control. you can keep your blood pressure under control by eating a healthy diet low in saturated fat, exercising regularly and, if needed, taking medicine to lower your blood pressure.\n")
                        st.write("7. Keep your Diabetes under control. There is a higher chance of developing Heart disease problems if you have diabetes.\n")
                        st.write("8. Manage stress.\n")
                        st.write("9. Get regular health screenings.")
                        medical_temp = """ 
                        <p style ="color:green;text-align:center;">Medical Recommendations</p>
                        """
                        st.markdown(medical_temp,unsafe_allow_html=True)
                        st.write("1. Go for Medical Checkups regularly.\n")
                        st.write("2. Take any prescribed medicine. In many occasions your doctor may prescribe medicine to prevent you developing heart-related problems.\n")
                        st.write("3. Consult the Doctors.")
                else:
                    st.success("The patient has No Heart Disease Related Problems")
                    #st.balloons()
                    st.snow()
                
if __name__ == '__main__':main()
           
                
                
                
                
                
            
        
        
            
            
        
    

                 