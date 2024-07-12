from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    requrements=[]
    with open(file_path) as file_obj:
        requrements=file_obj.readlines()
        requrements=[req.replace("\n","")for req in requrements]
        if HYPEN_E_DOT in requrements:
            requrements.remove(HYPEN_E_DOT)
        return requrements
setup(
    name='DiamondPricePrediction',
    version='0.0.1',
    author='Rohan',
    author_email='waghulkarrohan0@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)