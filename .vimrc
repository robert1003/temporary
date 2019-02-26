call plug#begin('~/.vim/plugged')

Plug 'itchyny/lightline.vim'
Plug 'crusoexia/vim-monokai'
Plug 'terryma/vim-multiple-cursors'
Plug 'tpope/vim-surround'

call plug#end()

syntax on
colorscheme monokai

set t_Co=256
set laststatus=2

se ai nu relativenumber ru cul
se cin et ts=2 sw=2 sts=2
so $VIMRUNTIME/mswin.vim
se gfn=Monospace\ 14



