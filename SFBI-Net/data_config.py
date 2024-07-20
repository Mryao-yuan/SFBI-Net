
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    image_type=""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '/data0/yy_data/Dataset/LEVIR-CD-256'
        elif data_name == 'DSIFN':
            self.label_transform = "without_norm"
            self.root_dir = '/data0/yy_data/Dataset/DSIFN-256'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = '/data/pth/yaoyuan-pth/Dataset/LEVIR-CD/'
        elif data_name == 'EGY_BCD':
            self.label_transform = "norm"
            self.root_dir = '/data0/yy_data/Dataset/EGY_BCD-unified-format/256/'
        elif data_name == 'EGY':
            self.label_transform = "norm"
            self.root_dir = '/data0/yy_data/Dataset/EGY_BCD'
        elif data_name =='CLCD-256':
            self.label_transform = "norm"
            self.root_dir = '/data0/yy_data/Dataset/CLCD-256/'
        elif data_name =='CLCD':
            self.label_transform = "norm"
            self.root_dir = '/data0/yy_data/Dataset/CLCD-256-Fin/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='CLCD-256')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

