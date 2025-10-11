import tkinter as tk
from tkinter import ttk, messagebox, font
import threading
import re
from main import initialize_search_engine, search, documents

class ModernSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Search Engine")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f4f8")
        
        # Color scheme
        self.colors = {
            'primary': '#6366f1',
            'primary_dark': '#4f46e5',
            'secondary': '#ec4899',
            'bg': '#f0f4f8',
            'card_bg': '#ffffff',
            'text': '#1e293b',
            'text_light': '#64748b',
            'border': '#e2e8f0',
            'hover': '#f8fafc',
            'success': '#10b981',
            'error': '#ef4444',
            'highlight': '#fef3c7',  # Yellow highlight for query terms
            'highlight_border': '#fbbf24'
        }
        
        self.is_initialized = False
        self.setup_styles()
        self.create_widgets()
        self.initialize_engine()
        
    def setup_styles(self):
        """Configure ttk styles for modern appearance."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Search button style
        style.configure('Search.TButton',
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 11, 'bold'),
                       padding=(20, 10))
        style.map('Search.TButton',
                 background=[('active', self.colors['primary_dark']),
                           ('pressed', self.colors['primary_dark'])])
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['card_bg'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_inner = tk.Frame(header_frame, bg=self.colors['card_bg'])
        header_inner.pack(fill=tk.X, padx=2, pady=2)
        
        title_label = tk.Label(header_inner,
                              text="🔍 Smart Document Search",
                              font=('Arial', 24, 'bold'),
                              bg=self.colors['card_bg'],
                              fg=self.colors['primary'])
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(header_inner,
                                 text="✓ Spell Correction  •  ✓ Query Highlighting",
                                 font=('Arial', 10),
                                 bg=self.colors['card_bg'],
                                 fg=self.colors['text_light'])
        subtitle_label.pack(pady=(0, 10))
        
        # Status label
        self.status_label = tk.Label(header_inner,
                                     text="Initializing...",
                                     font=('Arial', 10),
                                     bg=self.colors['card_bg'],
                                     fg=self.colors['text_light'])
        self.status_label.pack(pady=(0, 10))
        
        # Search section
        search_frame = tk.Frame(main_frame, bg=self.colors['card_bg'])
        search_frame.pack(fill=tk.X, pady=(0, 20))
        
        search_inner = tk.Frame(search_frame, bg=self.colors['card_bg'])
        search_inner.pack(fill=tk.X, padx=20, pady=20)
        
        # Search input container
        input_container = tk.Frame(search_inner, bg=self.colors['card_bg'])
        input_container.pack(fill=tk.X)
        
        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(input_container,
                                     textvariable=self.search_var,
                                     font=('Arial', 14),
                                     bg='white',
                                     fg=self.colors['text'],
                                     relief=tk.SOLID,
                                     borderwidth=2,
                                     insertbackground=self.colors['primary'])
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, ipadx=10)
        self.search_entry.bind('<Return>', lambda e: self.perform_search())
        self.search_entry.configure(highlightthickness=2, 
                                   highlightbackground=self.colors['border'],
                                   highlightcolor=self.colors['primary'])
        
        # Search button
        self.search_btn = ttk.Button(input_container,
                                     text="Search",
                                     style='Search.TButton',
                                     command=self.perform_search)
        self.search_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Results label
        self.results_label = tk.Label(main_frame,
                                     text="",
                                     font=('Arial', 12),
                                     bg=self.colors['bg'],
                                     fg=self.colors['text_light'])
        self.results_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Results container with scrolling
        results_container = tk.Frame(main_frame, bg=self.colors['card_bg'])
        results_container.pack(fill=tk.BOTH, expand=True)
        
        canvas_frame = tk.Frame(results_container, bg=self.colors['card_bg'])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.canvas = tk.Canvas(canvas_frame, bg=self.colors['card_bg'], 
                               highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", 
                                 command=self.canvas.yview)
        
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.colors['card_bg'])
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Show initial empty state
        self.show_empty_state()
        
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def initialize_engine(self):
        """Initialize search engine in background thread."""
        def init_thread():
            dataset_file = "Articles.csv"
            success = initialize_search_engine(dataset_file)
            self.root.after(0, self._on_init_complete, success, len(documents))
        
        thread = threading.Thread(target=init_thread, daemon=True)
        thread.start()
        
    def _on_init_complete(self, success, doc_count):
        """Handle initialization completion."""
        if success:
            self.is_initialized = True
            self.status_label.config(
                text=f"✓ Ready - {doc_count} documents loaded",
                fg=self.colors['success']
            )
            self.search_entry.config(state='normal')
            self.search_btn.config(state='normal')
        else:
            self.status_label.config(
                text="✗ Failed to load documents",
                fg=self.colors['error']
            )
            messagebox.showerror(
                "Initialization Error",
                "Could not load documents. Please check that 'Articles.csv' exists."
            )
        
    def show_empty_state(self):
        """Display empty state message."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        empty_frame = tk.Frame(self.scrollable_frame, bg=self.colors['card_bg'])
        empty_frame.pack(fill=tk.BOTH, expand=True, pady=50)
        
        icon_label = tk.Label(empty_frame,
                            text="🔎",
                            font=('Arial', 48),
                            bg=self.colors['card_bg'])
        icon_label.pack(pady=(20, 10))
        
        msg_label = tk.Label(empty_frame,
                           text="Start your search",
                           font=('Arial', 18, 'bold'),
                           bg=self.colors['card_bg'],
                           fg=self.colors['text'])
        msg_label.pack()
        
        sub_label = tk.Label(empty_frame,
                           text="Enter a query above - typos will be auto-corrected!",
                           font=('Arial', 12),
                           bg=self.colors['card_bg'],
                           fg=self.colors['text_light'])
        sub_label.pack(pady=(5, 20))
        
    def perform_search(self):
        """Execute search query."""
        if not self.is_initialized:
            messagebox.showwarning("Not Ready", "Search engine is still initializing...")
            return
            
        query = self.search_var.get().strip()
        if not query:
            return
        
        # Show loading state
        self.results_label.config(text="Searching...")
        self.search_btn.config(state='disabled')
        self.root.update()
        
        # Run search in thread
        thread = threading.Thread(target=self._search_thread, args=(query,), daemon=True)
        thread.start()
        
    def _search_thread(self, query):
        """Execute search in background thread."""
        try:
            results = search(query, top_n=5)
            self.root.after(0, self.display_results, query, results)
        except Exception as e:
            self.root.after(0, self._on_search_error, str(e))
        finally:
            self.root.after(0, lambda: self.search_btn.config(state='normal'))
    
    def _on_search_error(self, error_msg):
        """Handle search errors."""
        self.results_label.config(text="An error occurred during search")
        messagebox.showerror("Search Error", f"Error: {error_msg}")
        
    def display_results(self, query, results):
        """Display search results."""
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Update results label
        if results:
            self.results_label.config(
                text=f'Search results for "{query}" - {len(results)} documents found'
            )
        else:
            self.results_label.config(text=f'No results found for "{query}"')
        
        if not results:
            self.show_no_results()
            return
        
        # Display each result
        for i, (doc_id, score, content, highlighted_content, query_terms) in enumerate(results):
            self.create_result_card(doc_id, score, content, highlighted_content, query_terms, i)
    
    def show_no_results(self):
        """Display no results message."""
        no_results_frame = tk.Frame(self.scrollable_frame, bg=self.colors['card_bg'])
        no_results_frame.pack(fill=tk.BOTH, expand=True, pady=50)
        
        icon_label = tk.Label(no_results_frame,
                            text="❌",
                            font=('Arial', 48),
                            bg=self.colors['card_bg'])
        icon_label.pack(pady=(20, 10))
        
        msg_label = tk.Label(no_results_frame,
                           text="No results found",
                           font=('Arial', 16, 'bold'),
                           bg=self.colors['card_bg'],
                           fg=self.colors['text'])
        msg_label.pack()
        
        sub_label = tk.Label(no_results_frame,
                           text="Try different search terms",
                           font=('Arial', 12),
                           bg=self.colors['card_bg'],
                           fg=self.colors['text_light'])
        sub_label.pack(pady=(5, 0))
    
    def create_highlighted_text(self, parent, text, bg_color):
        """
        Create a text widget with highlighted query terms.
        
        Args:
            parent: Parent widget
            text: Text content with __term__ markers for highlighting
            bg_color: Background color
        """
        # Create text widget
        text_widget = tk.Text(parent,
                             wrap=tk.WORD,
                             font=('Arial', 11),
                             bg=bg_color,
                             fg=self.colors['text_light'],
                             relief=tk.FLAT,
                             borderwidth=0,
                             height=6,
                             cursor="arrow")
        
        # Configure tags for highlighting
        text_widget.tag_configure("highlight", 
                                 background=self.colors['highlight'],
                                 foreground=self.colors['text'],
                                 font=('Arial', 11, 'bold'))
        
        # Parse and insert text with highlights
        pattern = re.compile(r'__(.*?)__')
        last_end = 0
        
        for match in pattern.finditer(text):
            # Insert text before highlight
            if match.start() > last_end:
                text_widget.insert(tk.END, text[last_end:match.start()])
            
            # Insert highlighted text
            highlighted_word = match.group(1)
            text_widget.insert(tk.END, highlighted_word, "highlight")
            
            last_end = match.end()
        
        # Insert remaining text
        if last_end < len(text):
            text_widget.insert(tk.END, text[last_end:])
        
        # Make text read-only
        text_widget.config(state=tk.DISABLED)
        
        return text_widget
            
    def create_result_card(self, doc_id, score, content, highlighted_content, query_terms, index):
        """Create a result card widget with highlighted query terms."""
        # Card container with shadow effect
        card_outer = tk.Frame(self.scrollable_frame, 
                             bg=self.colors['border'])
        card_outer.pack(fill=tk.X, padx=15, pady=8)
        
        card = tk.Frame(card_outer, bg='white')
        card.pack(fill=tk.X, padx=1, pady=1)
        
        # Card content
        content_frame = tk.Frame(card, bg='white')
        content_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Header with title and score
        header_frame = tk.Frame(content_frame, bg='white')
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Document icon and title
        title_frame = tk.Frame(header_frame, bg='white')
        title_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        icon_label = tk.Label(title_frame,
                            text="📄",
                            font=('Arial', 16),
                            bg='white')
        icon_label.pack(side=tk.LEFT, padx=(0, 8))
        
        title_label = tk.Label(title_frame,
                              text=doc_id,
                              font=('Arial', 14, 'bold'),
                              bg='white',
                              fg=self.colors['text'],
                              anchor=tk.W)
        title_label.pack(side=tk.LEFT, fill=tk.X)
        
        # Score badge
        score_frame = tk.Frame(header_frame, bg=self.colors['primary'])
        score_frame.pack(side=tk.RIGHT)
        
        score_label = tk.Label(score_frame,
                              text=f"Score: {score:.4f}",
                              font=('Arial', 10, 'bold'),
                              bg=self.colors['primary'],
                              fg='white',
                              padx=12,
                              pady=4)
        score_label.pack()
        
        # Highlighted content with Text widget
        display_text = highlighted_content if len(highlighted_content) <= 400 else highlighted_content[:397] + "..."
        
        text_widget = self.create_highlighted_text(content_frame, display_text, 'white')
        text_widget.pack(fill=tk.X, pady=(5, 0))
        
        # Query terms info
        if query_terms:
            terms_frame = tk.Frame(content_frame, bg='white')
            terms_frame.pack(fill=tk.X, pady=(10, 0))
            
            terms_label = tk.Label(terms_frame,
                                  text=f"🔍 Matched terms: {', '.join(query_terms)}",
                                  font=('Arial', 9, 'italic'),
                                  bg='white',
                                  fg=self.colors['text_light'])
            terms_label.pack(anchor=tk.W)

def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = ModernSearchGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()