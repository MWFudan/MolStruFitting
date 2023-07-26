# MolStruFitting
Rotational constant based fitting program

## Description

Program used to fit structure to have similar rotational constant with the given value. It is designed to generate possible conformers of cluster and accelerate structure determination workflow. 
Original publication: doi:10.1063/1674-0068/cjcp2304042
![image](https://github.com/MWFudan/MolStruFitting/blob/main/img/%E5%9B%BE%E7%89%872.png)

## Getting Started

### Dependencies

* Python 3 with numpy, scipy, tqdm package
* Tested in Win11 and Ubuntu20

### Installing

* Just download files and run with python
* If MTCR sample not work in Linux, change the `random` method as in comments

### Executing program


Change parameters in `__name__ == '__main__'` block and run it. The parameters are explained in comments

The output file contains all possible candidates, which can be ranked using XTB method (doi: 10.1021/acs.jctc.7b00118). To rank these candidates using GFN1-xtb method, install xtb-python, download scripts in XTB_sort directory and run

```
python xtb_sort.py XXX
```

When finish, a file containing ranked candidates is available. 

Note: xtb sort method is only accessible in Linux.
## Authors


* Xinlei Chen, 20110220046@fudan.edu.cn

* Weixin Li, weixingli@fudan.edu.cn


## Acknowledgments

* [awesome-readme](https://github.com/matiassingers/awesome-readme)
