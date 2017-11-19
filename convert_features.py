
import pandas as pd
import numpy as np

def main():
    data = pd.read_csv('adult.test.txt', header = None)
    data.columns = ["age", "work class", "fnlwgt", "education", "education-num",
                    "marital-status", "occupation", "relationship", "race", "sex",
                    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]


    # original data
    
    #########################################################
    #  delete column 'fnlwgt' 'education-num' 'relationship'#

   # del data['fnlwgt']
    del data['education']
    del data['relationship']

    # 'fnlwgt' - hard to interpretate                       #
    # 'education-num' - "duplicates" of 'education'         #
    # 'relationship' - "duplicates" of 'marital-status'     #
    #########################################################
    
    ###############################
    #  convert column work class  #
    
    data['work class'] = data['work class'].str.replace('Private', '5')
    data['work class'] = data['work class'].str.replace('Self-emp-not-inc', '3')
    data['work class'] = data['work class'].str.replace('Self-emp-inc', '4')
    data['work class'] = data['work class'].str.replace('Federal-gov', '8')
    data['work class'] = data['work class'].str.replace('Local-gov', '6')
    data['work class'] = data['work class'].str.replace('State-gov', '7')
    data['work class'] = data['work class'].str.replace('Without-pay', '2')
    data['work class'] = data['work class'].str.replace('Never-worked', '1')
    data['work class'] = data['work class'].str.replace('?', '0')

    #  convert column work class  #
    ###############################
    # Private                 - 1 #
    # Self-emp-not-inc        - 2 #
    # Self-emp-inc            - 3 #
    # Federal-gov             - 4 #
    # Local-gov               - 5 #
    # State-gov               - 6 #
    # Without-pay             - 7 #
    # Never-worked            - 8 #
    # ?                       - 0 #
    ###############################

    

    ##########################
    #  convert column race   #
    
    data['race'] = data['race'].str.replace('White', '1') 
    data['race'] = data['race'].str.replace('Black', '2')
    data['race'] = data['race'].str.replace('Asian-Pac-Islander', '3')
    data['race'] = data['race'].str.replace('Amer-Indian-Eskimo', '4')
    data['race'] = data['race'].str.replace('Other', '5')
    data['race'] = data['race'].str.replace('?', '0')
    
    #  convert column race   #
    ##########################
    # White              - 1 #
    # Black              - 2 #
    # Asian-Pac-Islander - 3 #
    # Amer-Indian-Eskimo - 4 #
    # Other              - 5 #
    # ?                  - 0 #
    ##########################

    ###################################
    #  convert column marital-status  #
    
    data['marital-status'] = data['marital-status'].str.replace('Married-civ-spouse', '7')
    data['marital-status'] = data['marital-status'].str.replace('Divorced', '2')
    data['marital-status'] = data['marital-status'].str.replace('Never-married', '1')
    data['marital-status'] = data['marital-status'].str.replace('Separated', '4')
    data['marital-status'] = data['marital-status'].str.replace('Widowed', '3')
    data['marital-status'] = data['marital-status'].str.replace('Married-spouse-absent', '5')
    data['marital-status'] = data['marital-status'].str.replace('Married-AF-spouse', '6')
    data['marital-status'] = data['marital-status'].str.replace('?', '0')

    #  convert column marital-status  #
    ###################################
    # Married-civ-spouse          - 1 #
    # Divorced                    - 2 #
    # Never-married               - 3 #
    # Separated                   - 4 #
    # Widowed                     - 5 #
    # Married-spouse-absent       - 6 #
    # Married-AF-spouse           - 7 #
    # ?                           - 0 #
    ###################################

    ###############################
    #  convert column occupation  #
    
    data['occupation'] = data['occupation'].str.replace('Tech-support', '1')
    data['occupation'] = data['occupation'].str.replace('Craft-repair', '2')
    data['occupation'] = data['occupation'].str.replace('Other-service', '3')
    data['occupation'] = data['occupation'].str.replace('Sales', '4')
    data['occupation'] = data['occupation'].str.replace('Exec-managerial', '5')
    data['occupation'] = data['occupation'].str.replace('Prof-specialty', '6')
    data['occupation'] = data['occupation'].str.replace('Handlers-cleaners', '7')
    data['occupation'] = data['occupation'].str.replace('Machine-op-inspct', '8')
    data['occupation'] = data['occupation'].str.replace('Adm-clerical', '9')
    data['occupation'] = data['occupation'].str.replace('Farming-fishing', '10')
    data['occupation'] = data['occupation'].str.replace('Transport-moving', '11')
    data['occupation'] = data['occupation'].str.replace('Priv-house-serv', '12')
    data['occupation'] = data['occupation'].str.replace('Protective-serv', '13')
    data['occupation'] = data['occupation'].str.replace('Armed-Forces', '14')
    data['occupation'] = data['occupation'].str.replace('?', '0')

    #  convert column occupation  #
    ###############################
    # Tech-support           - 1  #
    # Craft-repair           - 2  #
    # Other-service          - 3  #
    # Sales                  - 4  #
    # Exec-managerial        - 5  #
    # Prof-specialty         - 6  #
    # Handlers-cleaners      - 7  #
    # Machine-op-inspct      - 8  #
    # Adm-clerical           - 9  #
    # Farming-fishing        - 10 #
    # Transport-moving       - 11 #
    # Priv-house-serv        - 12 #
    # Protective-serv        - 13 #
    # Armed-Forces           - 14 #
    # ?                      - 0  #
    ###############################

    ###############################
    #     convert column sex      #

    data['sex'] = data['sex'].str.replace('Female', '1')
    data['sex'] = data['sex'].str.replace('Male', '2')
    data['sex'] = data['sex'].str.replace('?', '2')

    #  convert column sex         #
    ###############################
    # Female                 - 0  #
    # Male                   - 1  #
    ###############################


    ##################################
    #  convert column native-counry  #

    data['native-country'] = data['native-country'].str.replace('United-States', '1')
    data['native-country'] = data['native-country'].str.replace('Cambodia', '2')
    data['native-country'] = data['native-country'].str.replace('England', '3')
    data['native-country'] = data['native-country'].str.replace('Puerto-Rico', '4')
    data['native-country'] = data['native-country'].str.replace('Canada', '5')
    data['native-country'] = data['native-country'].str.replace('Germany', '6')
    data['native-country'] = data['native-country'].str.replace('Outlying-US(Guam-USVI-etc)', '7')
    data['native-country'] = data['native-country'].str.replace('India', '8')
    data['native-country'] = data['native-country'].str.replace('Japan', '9')
    data['native-country'] = data['native-country'].str.replace('Greece', '10')
    data['native-country'] = data['native-country'].str.replace('South', '11')
    data['native-country'] = data['native-country'].str.replace('China', '12')
    data['native-country'] = data['native-country'].str.replace('Cuba', '13')
    data['native-country'] = data['native-country'].str.replace('Iran', '14')
    data['native-country'] = data['native-country'].str.replace('Honduras', '15')
    data['native-country'] = data['native-country'].str.replace('Philippines', '16')
    data['native-country'] = data['native-country'].str.replace('Italy', '17')
    data['native-country'] = data['native-country'].str.replace('Poland', '18')
    data['native-country'] = data['native-country'].str.replace('Jamaica', '19')
    data['native-country'] = data['native-country'].str.replace('Vietnam', '20')
    data['native-country'] = data['native-country'].str.replace('Mexico', '21')
    data['native-country'] = data['native-country'].str.replace('Portugal', '22')
    data['native-country'] = data['native-country'].str.replace('Ireland', '23')
    data['native-country'] = data['native-country'].str.replace('France', '24')
    data['native-country'] = data['native-country'].str.replace('Dominican-Republic', '25')
    data['native-country'] = data['native-country'].str.replace('Laos', '26')
    data['native-country'] = data['native-country'].str.replace('Ecuador', '27')
    data['native-country'] = data['native-country'].str.replace('Taiwan', '28')
    data['native-country'] = data['native-country'].str.replace('Haiti', '29')
    data['native-country'] = data['native-country'].str.replace('Columbia', '30')
    data['native-country'] = data['native-country'].str.replace('Hungary', '31')
    data['native-country'] = data['native-country'].str.replace('Guatemala', '32')
    data['native-country'] = data['native-country'].str.replace('Nicaragua', '33')
    data['native-country'] = data['native-country'].str.replace('Scotland', '34')
    data['native-country'] = data['native-country'].str.replace('Thailand', '35')
    data['native-country'] = data['native-country'].str.replace('Yugoslavia', '36')
    data['native-country'] = data['native-country'].str.replace('El-Salvador', '37')
    data['native-country'] = data['native-country'].str.replace('Trinadad&Tobago', '38')
    data['native-country'] = data['native-country'].str.replace('Peru', '39')
    data['native-country'] = data['native-country'].str.replace('Hong', '40')
    data['native-country'] = data['native-country'].str.replace('Holand-Netherlands', '41')
    data['native-country'] = data['native-country'].str.replace('?', '0')
    
    #  convert column native-counry  #
    ##################################

    ##################################
    #     convert column income      #
    data['income'] = data['income'].str.replace('>50K', '1')
    data['income'] = data['income'].str.replace('<=50K', '0')
    data['income'] = data['income'].str.replace('?', '2')
    #     convert column income      #
    ##################################

    np.savetxt(r'c:\Python27\converted_test_adult.txt', data.values, fmt='%s')
    

if __name__ == "__main__":
    main()

