function varargout = CalmXpert(varargin)
% CalmXpert MATLAB code for CalmXpert.fig
%      CalmXpert, by itself, creates a new CalmXpert or raises the existing
%      singleton*.
%
%      H = CalmXpert returns the handle to a new CalmXpert or the handle to
%      the existing singleton*.
%
%      CalmXpert('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CalmXpert.M with the given input arguments.
%
%      CalmXpert('Property','Value',...) creates a new CalmXpert or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CalmXpert_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CalmXpert_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CalmXpert

% Last Modified by GUIDE v2.5 10-Apr-2019 14:09:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CalmXpert_OpeningFcn, ...
                   'gui_OutputFcn',  @CalmXpert_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CalmXpert is made visible.
function CalmXpert_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CalmXpert (see VARARGIN)

% Choose default command line output for CalmXpert
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CalmXpert wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CalmXpert_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
Fs = 48000;
nBits = 8;
nChannels = 1;
deviceID = 1;
handles.rec.recObj = audiorecorder(Fs, nBits, nChannels, deviceID);
guidata(hObject,handles)
record(handles.rec.recObj);
set(handles.pushbutton2, 'Enable', 'on');
uicontrol(handles.pushbutton2);




% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.pushbutton1, 'Enable', 'off');
recObj = handles.rec.recObj;
get(recObj);
stop(recObj);
myRecording = getaudiodata(recObj);
axes(handles.axes1);
Rec_min = repmat(min(myRecording),size(myRecording),1);
ind_Rec = myRecording - Rec_min;
Rec_max=repmat(max(ind_Rec),size(ind_Rec,1),1);
Norm_Rec = ind_Rec./Rec_max;
plot(Norm_Rec);
drawnow
set(handles.text_ok, 'String', 'Recording completed')
set(handles.text_ok, 'ForegroundColor', 'g')
set(handles.text_ok, 'FontSize', 16)
axes(handles.axes2)
spectrogram(myRecording(:,1), 'yaxis')
load ('deepnet_1');
Fs = 48000;
startSample = 1.0*Fs;
endSample = 3.12*Fs;
Inp_inp = Norm_Rec(startSample:endSample);
len=1e3;
Nfix=fix(size(Inp_inp,1)/len);
Input_data_test = reshape(Inp_inp(1:Nfix*len,:),len,[]);
xmin_t = repmat(min(Input_data_test),size(Input_data_test,1),1);
Input_1 = Input_data_test - xmin_t;
xmax_t=repmat(max(Input_1),size(Input_1,1),1);
Input = Input_1./xmax_t;
y_test = sim(deepnet_1,Input);
N_test_an = 0;
N_test_oth = 0;
for i = 1:length(y_test)
    if y_test(1,i) > 0.5
        N_test_an = N_test_an + 1;
    else
         N_test_an = N_test_an;
    end
end
for j = 1:1:length(y_test)
    if y_test(2,j) > 0.67
        N_test_oth = N_test_oth + 1;
    else
        N_test_oth = N_test_oth;
    end
end
%if N_test_an > N_test_oth
    set(handles.text_rec, 'String', 'Agressive')
    set(handles.text_rec, 'ForegroundColor', 'r')
    set(handles.text_rec, 'FontSize', 16)
%else
%    set(handles.text_rec, 'String', 'Not Agressive')
%    set(handles.text_rec, 'ForegroundColor', 'g')
%    set(handles.text_rec, 'FontSize', 16)
%end
set(handles.pushbutton2, 'Enable', 'off');
set(handles.pushbutton1, 'Enable', 'on');
uicontrol(handles.pushbutton1);
