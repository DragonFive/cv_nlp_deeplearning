# -*- coding: gbk -*-
#__author__ = 'ASUS'

import time,sys, urllib, urllib2, json,socket,re

# 新建一个python文件
def newPy(fname):
    fileName=r"E:\资料\onedrive\code\python\opencv"+"\\"+fname;
    f = open(fileName, "a");
    f.write("# -*- coding: gbk -*-\n#__author__ = 'dragonfive'\n");
    # 获取当前时间
    now_clock = getSysTime()
    f.write("# "+now_clock+"\n")
    # 获取当前地点
    myip = Getmyip()
    address = myip.getAddByIp(myip.get_ex_ip()).encode('gbk');
    f.write("# "+address+" dragonfive \n");
    # 联系方式
    f.write("# any question mail：dragonfive1992@gmail.com \n");
    # 版权信息
    f.write("# copyright 1992-"+time.strftime(r"%Y")+ " dragonfive \n");
    f.close();

# 获取系统时间
def getSysTime():
    now_clock = time.strftime(r"%d/%m/%Y %H:%M:%S")
    return now_clock

# 获取百度ipstore的apikey
def getMyApikey():
    return "9b447786edf207a8813e115d44d85187"


class Getmyip:
    # 获得本机内网IP
    def get_local_ip(self):
        localIP=socket.gethostbyname(socket.gethostname())
        return localIP;

    # 获取外网IP
    def get_ex_ip(self):
        try:
            myip = self.visit("http://www.whereismyip.com/")
        except:
            try:
                myip = self.visit("http://www.ip138.com/ip2city.asp")
            except:
                myip = "So sorry!!!"
        return myip

    def visit(self,url):
        opener = urllib2.urlopen(url)
        ourl=opener.geturl()
        if url == ourl:
            str = opener.read()
            asd=re.search('\d+\.\d+\.\d+\.\d+',str).group(0)
        return asd

    # 获取一个ip的区域地址，用百度ipstore的工具;
    def getAddByIp(self,ipaddr):#这里可以用多参数进行优化；
        url = 'http://apis.baidu.com/chazhao/ipsearch/ipsearch?ip='+ipaddr
        req = urllib2.Request(url)

        req.add_header("apikey", getMyApikey());

        resp = urllib2.urlopen(req)
        content = json.loads(resp.read())
        if(content):
            address=content["data"]["country"]+":"+content["data"]["city"]+":"+content["data"]["operator"]
            return address;

def is_huiwenshu(n):
    return n==int(''.join(list(reversed(str(n)))))


def get_huiwenshu():
    it=range(100,1000)
    it = filter(is_huiwenshu,it)
    print([i for i in it if i<1000])

