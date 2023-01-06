#Python imports
import streamlit as st
import pandas as pd
import numpy as np
import sys

#Other packages
from tool import *
from poly import Builder
from functions_to_use import *

#Setting tab icons and name
st.set_page_config(page_title='Solver - 3', 
                   layout='wide')

#seting color theme 
st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fd0de;
    }
    </style>
    """, unsafe_allow_html=True)

#Setting general title 
st.title('Solver')

#Dividing page into three parts (main and parameters input + output) 
main, dims, degs, add = st.columns(4)

#Setting main input header
main.header('Files')

#Declared variables for input/output files will be used
input_name = main.file_uploader('Input file name', type=['csv'], key='input_file')
output_name = main.text_input('Output file name', value='output', key='output_file')

#Setting header for dimension input 
dims.header('Input dimensionality')

#Declaring variables for dimensionality of data
dim = dims.number_input('Dimension of Y', value=4, step=1, key='dim')
dim_1 = dims.number_input('Dimension of X1', value=2, step=1, key='dim_1')
dim_2 = dims.number_input('Dimension of X2', value=2, step=1, key='dim_2')
dim_3 = dims.number_input('Dimension of X3', value=3, step=1, key='dim_3')

#Same for degrees
degs.header('Input polynoms degrees ')

#Declaring variables
degree_1 = degs.number_input('Degree for X1', value=0, step=1, key='degree_1')
degree_2 = degs.number_input('Degree for X2', value=0, step=1, key='degree_2')
degree_3 = degs.number_input('Degree for X3', value=0, step=1, key='degree_3')

#Additional input section, some specifications
add.header('Input other parameters ')
use_type = add.radio('Polynomial type used: ', ['Chebyshev', 'Chebyshev shifted'])
function_struct = add.checkbox('Enable tanh function')
normalize = add.checkbox('Plot normalized plots ')

#Defining functionality of run button
if main.button('Run', key='run'):
    try:
        #try-block
        #Parsing file recieved
        try:
            input_file_text = str(input_name.getvalue().decode())
        except:
            #default variant
            input_file_text = """5,05;8,65;7,75;6,975;4,879;3,501;5,967;254,621;98,145;119,406;117,683;5,052;8,7;7,78;6,955;4,886;3,553;5,978;198,163;73,368;92,651;90,123;5,055;8,745;7,8;6,95;4,897;3,611;5,984;187,411;91,084;87,691;83,576;5,06;8,75;7,82;6,945;4,916;3,652;5,987;167,197;123,567;78,793;74,789;5,063;9,8;7,845;6,925;4,938;3,723;5,996;166,547;163,813;79,497;74,316;5,64;10,25;7,851;6,895;4,947;3,758;5,999;153,789;261,378;77,082;72,817;5,067;11,85;7,852;6,865;4,954;3,784;5,976;110,926;355,579;67,758;77,425;5,07;12,87;7,853;6,854;4,967;3,809;5,964;151,381;440,432;51,956;89,519;5,075;14,9;8,854;6,856;4,978;3,825;5,958;187,364;336,283;91,123;121,374;5,08;16,91;8,855;6,855;4,984;3,845;5,937;236,123;223,657;112,859;149,173;5,085;18,92;9,856;6,856;4,987;3,851;5,916;292,341;118,624;153,717;184,136;5,09;15,92;10,86;6,865;4,996;3,8534;5,874;344,324;91,324;117,965;179,152;5,095;12,93;11,85;7,859;4,999;3,8536;5,842;426,939;68,926;155,912;201,239;5,1;11,93;12,87;7,876;4,976;3,854;5,814;477,128;44,675;169,359;225,482;5,125;9,935;11,89;7,895;4,964;3,856;5,756;505,327;29,367;192,924;240,976;5,135;8,941;9,925;7,925;4,958;3,859;5,718;558,386;18,567;218,549;275,846;5,15;7,945;8,945;7,945;4,937;3,867;5,671;618,859;23,932;247,354;316,124;5,153;6,951;7,945;7,951;4,916;3,879;5,629;895,737;35,124;284,167;363,928;5,157;5,965;6,95;6,955;4,874;3,886;5,567;906,168;61,946;316,375;403,153;5,2;4,965;5,965;6,975;4,842;3,897;5,486;885,761;121,387;341,326;431,195;5,25;3,974;4,975;7,001;4,814;3,916;5,452;790,639;310,519;375,651;471,588;5,3;2,981;5;7,125;4,756;3,938;5,501;723,784;485,142;446,856;436,847;5,315;3,985;6,975;7,145;4,718;3,947;5,554;731,438;588,125;548,314;441,842;5,325;4,99;7,955;7,165;4,671;3,954;5,621;721,321;683,435;644,716;439,425;5,35;5,995;8,945;7,195;4,629;3,967;5,658;691,845;772,834;729,942;422,147;5,353;7,997;9,935;7,209;4,567;3,978;5,712;508,614;880,562;849,316;435,954;5,357;9,001;10,92;7,225;4,482;3,984;5,753;429,956;687,987;748,231;450,492;5,4;10,94;11,89;7,25;4,452;3,987;5,781;330,129;488,951;647,987;454,897;5,425;12,9;12,86;7,975;4,364;3,996;5,802;127,152;385,494;442,967;458,289;5,445;10,88;14,85;7,955;4,326;3,999;5,825;78,654;211,209;232,856;172,164;5,465;8,944;15,85;7,95;4,264;3,976;5,845;52,145;196,197;115,632;153,356;5,475;6,78;12,85;7,945;4,184;3,964;5,851;86,243;87,325;93,135;127,168;5,85;6,764;10,85;7,925;4,156;3,958;5,854;126,345;64,615;77,824;106,123;5,495;6,568;8,865;7,895;4,136;3,937;5,856;132,879;52,534;63,453;82,659;5,497;6,437;6,859;7,865;4,129;3,916;5,854;167,156;32,178;52,167;93,834;5,5;5,325;4,876;7,854;4,116;3,874;5,856;170,531;66,176;42,836;91,345;5,515;5,206;2,895;7,853;4,098;3,842;5,859;184,243;70,364;37,192;96,841;5,525;5,149;1,925;7,855;4,0816;3,814;5,867;191,956;76,428;25,834;93,952;5,545;5,089;3,945;7,856;4,0686;3,756;5,879;216,829;83,475;50,985;109,463;5,575;4,933;4,953;7,865;4,0486;3,718;5,886;383,329;104,924;98,591;133,415;5,6;4,889;5,955;7,859;4,0246;3,671;5,005;279,421;184,183;102,861;108,613;5,65;3,935;6,975;7,876;4,0126;3,629;5,027;225,356;286,324;105,817;107,319;5,7;3,941;7,001;7,895;4,0114;3,567;5,049;176,578;366,457;78,473;82,263;5,745;2,945;7,125;7,925;4,0026;3,484;5,095;170,948;265,814;81,417;84,132;5,75;3,95;7,145;7,945;4,0019;3,452;5,189;158,334;184,549;78,653;81,953
"""
            
        input_file = input_file_text.replace(",",".").replace(';', '\t')
        
        #Storing parameters in convinient way
        params = {
            'dimensions': [dim_1, dim_2, dim_3, dim],
            'input_file': input_file,
            'output_file': output_name + '.csv',
            'degrees': [degree_1, degree_2, degree_3],
            'polynomial_type': use_type,
            'mode': function_struct*1
        }
      
        
        #Processing of data using packages created previously
        with st.spinner('...'):
            solver, degrees = get_solution(params, pbar_container=main, max_deg=7) 
      
        solution = Builder(solver) 

        #Showing and plotting errors
        error_cols = st.columns(2)
    
        for ind, info in enumerate(solver.show()[-2:]):
            error_cols[ind].subheader(info[0])
            error_cols[ind].dataframe(info[1])
        
        #Saving results in variables
        if normalize:
            Y_values = solution._solution.Y
            final_values = solution._solution.final
        else:
            #Saving results in variables
            Y_values = solution._solution.Y_
            final_values = solution._solution.final_d
            
       
        cols = Y_values.shape[1]
        
        #Results section
        st.subheader('Results')
        
        #Defining layout of plots
        plot_cols = st.columns(cols)
        
        #Plotting residuals, components for each dimension of Y
        for n in range(cols):
            df = pd.DataFrame(
                np.array([Y_values[:, n], final_values[:, n]]).T,
                columns=[f'Y{n+1}', f'F{n+1}']
            )
            plot_cols[n].write(f'Component №{n+1}')
            plot_cols[n].line_chart(df)
            plot_cols[n].write(f'Сomponent\'s №{n+1} residual')
            
            df = pd.DataFrame(
                np.abs(Y_values[:, n] - final_values[:, n]).T,
                columns=[f'Error{n+1}']
            )
            plot_cols[n].line_chart(df)
        
        #Show polynoms
        matrices = solver.show()[:-2]
                                 
        if normalize:
            st.subheader(matrices[1][0])
            st.dataframe(matrices[1][1])
        else:
            st.subheader(matrices[0][0])
            st.dataframe(matrices[0][1])

        st.write(solution.get_results())

        matr_cols = st.columns(3)
        
        for ind, info in enumerate(matrices[2:5]):
            matr_cols[ind].subheader(info[0])
            matr_cols[ind].dataframe(info[1])
        
        #Downloading output button
        with open(params['output_file'], 'rb') as fout:
            main.download_button(
                label='Download output file',
                data=fout,
                file_name=params['output_file']
            )
            
    except Exception as ex:
        #except-block, if something goes wrong
        st.write("Exception :"+ str(sys.exc_info()) + ":: Check input and try again")