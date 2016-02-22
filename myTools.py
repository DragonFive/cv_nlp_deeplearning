# -*- coding: gbk -*-
#__author__ = 'ASUS'

import time,sys, urllib, urllib2, json,socket,re

# �½�һ��python�ļ�
def newPy(fname):
    fileName=r"E:\����\onedrive\code\python\opencv"+"\\"+fname;
    f = open(fileName, "a");
    f.write("# -*- coding: gbk -*-\n#__author__ = 'dragonfive'\n");
    # ��ȡ��ǰʱ��
    now_clock = getSysTime()
    f.write("# "+now_clock+"\n")
    # ��ȡ��ǰ�ص�
    myip = Getmyip()
    address = myip.getAddByIp(myip.get_ex_ip()).encode('gbk');
    f.write("# "+address+" dragonfive \n");
    # ��ϵ��ʽ
    f.write("# any question mail��dragonfive1992@gmail.com \n");
    # ��Ȩ��Ϣ
    f.write("# copyright 1992-"+time.strftime(r"%Y")+ " dragonfive \n");
    f.close();

# ��ȡϵͳʱ��
def getSysTime():
    now_clock = time.strftime(r"%d/%m/%Y %H:%M:%S")
    return now_clock

# ��ȡ�ٶ�ipstore��apikey
def getMyApikey():
    return "9b447786edf207a8813e115d44d85187"


class Getmyip:
    # ��ñ�������IP
    def get_local_ip(self):
        localIP=socket.gethostbyname(socket.gethostname())
        return localIP;

    # ��ȡ����IP
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

    # ��ȡһ��ip�������ַ���ðٶ�ipstore�Ĺ���;
    def getAddByIp(self,ipaddr):#��������ö���������Ż���
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

